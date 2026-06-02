"""
VLA Inference Client — runs in the AReaL training environment.

Drop-in replacement for VLALocalEngine.  The model runs in a separate
process (vla_inference_server.py) inside the SimpleVLA conda environment;
this client talks to it over ZMQ.

Multiple concurrent agenerate() calls are handled correctly: each call
registers a UUID → Future mapping, a background task resolves futures as
responses arrive, so all of AReaL's concurrent episode coroutines can have
requests in flight simultaneously without blocking each other.

Usage in libero_rl.py:
    from areal.engine.vla_inference_client import VLAInferenceClient

    engine = VLAInferenceClient("tcp://localhost:5555")
    await engine.connect()            # verify server is alive

    # ... training loop ...
    actor.save_checkpoint(ckpt_path)
    await engine.update_weights(ckpt_path, version=step + 1)

    await engine.close()
"""

from __future__ import annotations

import asyncio
import logging
import os
import pickle
import uuid
from typing import Any

import numpy as np

try:
    import zmq
    import zmq.asyncio
except ImportError as e:
    raise ImportError(
        "pyzmq is required for VLAInferenceClient.  "
        "Install it: pip install pyzmq"
    ) from e

from areal.workflow.vla_robot import VLAEngine, VLAStepRequest, VLAStepResponse

logger = logging.getLogger("VLAClient")


class VLAInferenceClient(VLAEngine):
    """
    Async ZMQ DEALER client that talks to VLAInferenceServer.

    Parameters
    ----------
    server_address : ZMQ connect string, e.g. "tcp://localhost:5555"
    ping_timeout   : seconds to wait for server PING on connect()
    request_timeout: seconds to wait for a GENERATE response
    """

    def __init__(
        self,
        server_address: str = "tcp://localhost:5555",
        ping_timeout: float = 60.0,
        request_timeout: float = 600.0,   # robot episodes can be slow
    ) -> None:
        self.server_address = server_address
        self.ping_timeout = ping_timeout
        self.request_timeout = request_timeout

        self._ctx: zmq.asyncio.Context | None = None
        self._socket: Any = None
        self._pending: dict[str, asyncio.Future] = {}
        self._recv_task: asyncio.Task | None = None
        self._version: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Connect and verify the server is alive. Call once before training."""
        self._ctx = zmq.asyncio.Context()
        self._socket = self._ctx.socket(zmq.DEALER)
        # Unique identity so ROUTER routes responses to this client only
        self._socket.identity = uuid.uuid4().bytes
        self._socket.connect(self.server_address)

        self._recv_task = asyncio.create_task(
            self._recv_loop(), name="vla_client_recv"
        )
        await self._ping()
        logger.info(
            f"Connected to VLA inference server at {self.server_address}  "
            f"(weight version={self._version})"
        )

    async def close(self) -> None:
        """Disconnect and clean up."""
        if self._recv_task:
            self._recv_task.cancel()
            try:
                await self._recv_task
            except asyncio.CancelledError:
                pass
        if self._socket:
            self._socket.close()
        if self._ctx:
            self._ctx.term()

    async def shutdown_server(self) -> None:
        """Tell the server process to exit."""
        await self._request({"type": "SHUTDOWN"}, timeout=10.0)

    # ------------------------------------------------------------------
    # VLAEngine interface
    # ------------------------------------------------------------------

    async def agenerate(self, req: VLAStepRequest) -> VLAStepResponse:
        """
        Send image + instruction to the server and return the response.

        The server returns a pre-decoded continuous action (using the model's
        norm_stats in the SimpleVLA env), so the workflow uses decoded_action
        directly for env.step() instead of calling the local action_decoder.
        """
        response = await self._request(
            {
                "type": "GENERATE",
                "image": req.image,             # numpy uint8, pickle-serialised
                "instruction": req.instruction_text or "",
                "action_chunks_len": req.max_new_tokens,
                "action_token_len": 7,           # fixed for OpenVLA-OFT
                "do_sample": False,              # greedy at training rollout time
                "temperature": 1.0,              # ignored when do_sample=False
            },
            timeout=self.request_timeout,
        )

        if response.get("status") != "ok":
            raise RuntimeError(
                f"VLAInferenceClient: server error — {response.get('message')}"
            )

        action_token_ids: list[int] = response["action_token_ids"]
        input_token_ids: list[int] = response["input_token_ids"]
        logprobs: list[float] = response["logprobs"]
        version: int = response.get("version", self._version)

        return VLAStepResponse(
            input_tokens=input_token_ids,
            output_tokens=action_token_ids,
            output_logprobs=logprobs,
            output_versions=[version] * len(action_token_ids),
            decoded_action=response.get("action"),  # continuous action from server
        )

    def get_version(self) -> int:
        return self._version

    # ------------------------------------------------------------------
    # Weight synchronisation
    # ------------------------------------------------------------------

    async def update_weights(
        self,
        checkpoint_path: str,
        version: int | None = None,
    ) -> None:
        """
        Tell the server to reload weights from a checkpoint file.

        Call this after actor.save_checkpoint() so rollout and training stay
        in sync.  The server loads from disk — no 14 GB transfer over the
        socket.

        Parameters
        ----------
        checkpoint_path : absolute path accessible from the server process
        version         : new weight version; defaults to current + 1
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(checkpoint_path)

        new_version = version if version is not None else self._version + 1
        response = await self._request(
            {
                "type": "RELOAD_WEIGHTS",
                "checkpoint_path": checkpoint_path,
                "version": new_version,
            },
            timeout=180.0,   # 7B model reload can take ~60s
        )

        if response.get("status") != "ok":
            raise RuntimeError(
                f"Weight reload failed: {response.get('message')}"
            )
        self._version = new_version
        logger.info(f"Weight sync complete — version={self._version}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _ping(self) -> None:
        response = await self._request({"type": "PING"}, timeout=self.ping_timeout)
        self._version = response.get("version", 0)

    async def _request(self, payload: dict, timeout: float = 60.0) -> dict:
        """Send a request and await the matching response."""
        request_id = str(uuid.uuid4())
        payload["request_id"] = request_id

        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[request_id] = future

        data = pickle.dumps(payload, protocol=4)
        # DEALER sends: [empty_delimiter, payload]
        await self._socket.send_multipart([b"", data])

        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            self._pending.pop(request_id, None)
            raise TimeoutError(
                f"VLAInferenceClient: no response in {timeout}s "
                f"(type={payload['type']})"
            )

    async def _recv_loop(self) -> None:
        """Background task: receive responses and resolve pending futures."""
        while True:
            try:
                parts = await self._socket.recv_multipart()
            except zmq.ZMQError:
                break

            # DEALER receives: [empty_delimiter, payload]
            payload = parts[-1]
            try:
                response = pickle.loads(payload)
            except Exception as exc:
                logger.warning(f"Deserialise failed: {exc}")
                continue

            req_id = response.get("request_id")
            if req_id and req_id in self._pending:
                future = self._pending.pop(req_id)
                if not future.done():
                    future.set_result(response)
