"""
VLA Inference Server — runs in the SimpleVLA conda environment.

This is a standalone process that owns the OpenVLA-OFT model and serves
inference requests from the AReaL training process over a ZMQ socket.
It has zero AReaL imports and works with Python 3.10 / PyTorch 2.2.

This bridges the dependency gap:
  SimpleVLA env  (Python 3.10, PyTorch 2.2, transformers-openvla-oft)
  AReaL env      (Python 3.12, PyTorch 2.9, FSDP2)

Launch (always from the SimpleVLA conda env):
    conda activate simplevla
    python areal/engine/vla_inference_server.py \
        --model_path Haozhan72/Openvla-oft-SFT-libero-spatial-traj1 \
        --benchmark libero_spatial \
        --address tcp://*:5555 \
        --device cuda:0

ZMQ protocol  (ROUTER socket, pickle serialisation):
    PING           → health check, returns current weight version
    GENERATE       → one VLA forward pass, returns action + token IDs
    RELOAD_WEIGHTS → reload model weights from a checkpoint path on disk
    SHUTDOWN       → graceful exit

Weight sync strategy:
    The AReaL actor calls actor.save_checkpoint(path) then sends
    RELOAD_WEIGHTS {"checkpoint_path": path, "version": N}.
    The server reloads from disk rather than receiving the full ~14 GB
    state dict over the socket.
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import pickle
import signal
import time
from typing import Any

import numpy as np
import torch
import zmq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("VLAServer")


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------


def center_crop_image(image: np.ndarray, scale: float = 0.95) -> np.ndarray:
    """
    Centre-crop then resize to original dimensions.
    Matches SimpleVLA-RL rob_rollout.py centre_crop_image() exactly.
    """
    from PIL import Image as PILImage  # type: ignore

    h, w = image.shape[:2]
    ch, cw = int(h * scale), int(w * scale)
    top, left = (h - ch) // 2, (w - cw) // 2
    cropped = image[top : top + ch, left : left + cw]
    return np.array(PILImage.fromarray(cropped).resize((w, h), PILImage.BILINEAR))


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------


class OpenVLAOFTWrapper:
    """
    Wraps OpenVLA-OFT and exposes a single generate() method.

    Mirrors _generate_one_step_oft() in SimpleVLA-RL rob_rollout.py:
      - centre-crops the image (if configured)
      - builds the prompt string
      - runs self.model.generate_action_verl()
      - returns continuous action + discrete token IDs
    """

    def __init__(
        self,
        model_path: str,
        unnorm_key: str,
        device: str = "cuda:0",
        center_crop: bool = True,
    ) -> None:
        from transformers import AutoModelForVision2Seq, AutoProcessor  # type: ignore

        logger.info(f"Loading model: {model_path}  device={device}")
        self.unnorm_key = unnorm_key
        self.center_crop = center_crop
        self.device = torch.device(device)

        self.model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )
        logger.info("Model loaded and ready")

    def generate(
        self,
        image: np.ndarray,
        instruction: str,
        action_chunks_len: int = 8,
        action_token_len: int = 7,
        do_sample: bool = False,
        temperature: float = 1.0,
    ) -> dict[str, Any]:
        """
        Run one VLA forward pass.

        Returns
        -------
        dict with:
            action           : np.ndarray (action_chunks_len, action_dim) float32
                               continuous robot action, already decoded via norm_stats
            action_token_ids : list[int]  discrete token IDs for trajectory tensors
            input_token_ids  : list[int]  prompt token IDs
            logprobs         : list[float] placeholder zeros
                               (AReaL recomputes from actor with recompute_logprob=True)
        """
        from PIL import Image as PILImage  # type: ignore

        if self.center_crop:
            image = center_crop_image(image)

        pil_image = PILImage.fromarray(image)
        prompt = f"In: What action should the robot take to {instruction}?\nOut:"

        inputs = self.processor(
            text=prompt,
            images=pil_image,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        pixel_values = inputs["pixel_values"].to(self.device, dtype=torch.bfloat16)

        total_tokens = action_token_len * action_chunks_len

        with torch.no_grad():
            # generate_action_verl is the custom method in transformers-openvla-oft.
            # Returns: (continuous_action, response_dict)
            actions, response = self.model.generate_action_verl(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                padding_idx=self.processor.tokenizer.pad_token_id,
                do_sample=do_sample,
                unnorm_key=self.unnorm_key,
                temperature=temperature,
            )

        # Normalise action shape → (action_chunks_len, action_dim)
        if isinstance(actions, torch.Tensor):
            actions = actions.squeeze(0).float().cpu().numpy()
        else:
            actions = np.array(actions, dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[np.newaxis, :]

        # Extract discrete token IDs generated after the prompt
        prompt_len = input_ids.shape[-1]
        all_ids = response["sequences"][0].cpu().tolist()
        action_token_ids = all_ids[prompt_len : prompt_len + total_tokens]
        input_token_ids = all_ids[:prompt_len]

        # Logprobs are placeholder zeros — AReaL's actor recomputes them
        # via recompute_logprob=True before the PPO update.
        logprobs = [0.0] * len(action_token_ids)

        return {
            "action": actions,
            "action_token_ids": action_token_ids,
            "input_token_ids": input_token_ids,
            "logprobs": logprobs,
        }

    def reload_weights(self, checkpoint_path: str) -> None:
        """Reload weights from a checkpoint saved by AReaL's actor."""
        logger.info(f"Reloading weights from {checkpoint_path}")
        state_dict = torch.load(
            checkpoint_path, map_location=self.device, weights_only=True
        )
        # Unwrap nested state dicts produced by different checkpoint formats
        for key in ("model", "state_dict", "module"):
            if key in state_dict and isinstance(state_dict[key], dict):
                state_dict = state_dict[key]
                break
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(f"  Missing keys  : {len(missing)}")
        if unexpected:
            logger.warning(f"  Unexpected keys: {len(unexpected)}")
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("Weight reload complete")


# ---------------------------------------------------------------------------
# ZMQ server
# ---------------------------------------------------------------------------


class VLAInferenceServer:
    """
    ZMQ ROUTER server.  Processes one GPU inference request at a time
    (the GPU can only run one forward pass at a time), but accepts requests
    from many concurrent client coroutines via the ROUTER/DEALER pattern.
    """

    def __init__(
        self,
        model: OpenVLAOFTWrapper,
        address: str = "tcp://*:5555",
    ) -> None:
        self._model = model
        self._version: int = 0
        self._running = True

        ctx = zmq.Context()
        self._socket = ctx.socket(zmq.ROUTER)
        self._socket.bind(address)
        logger.info(f"Bound to {address}")

        signal.signal(signal.SIGTERM, self._stop)
        signal.signal(signal.SIGINT, self._stop)

    def _stop(self, *_) -> None:
        logger.info("Shutdown signal received")
        self._running = False

    def run(self) -> None:
        logger.info("Server ready")
        while self._running:
            # 1-second poll so SIGTERM is processed promptly
            if not self._socket.poll(timeout=1000):
                continue

            # ROUTER frame layout: [identity, empty_delimiter, payload]
            parts = self._socket.recv_multipart()
            if len(parts) < 3:
                continue
            client_id, payload = parts[0], parts[2]

            try:
                req = pickle.loads(payload)
            except Exception as exc:
                logger.warning(f"Deserialise failed: {exc}")
                continue

            req_type = req.get("type", "")
            req_id = req.get("request_id", "")

            try:
                if req_type == "PING":
                    resp = {"type": "PONG", "request_id": req_id,
                            "status": "ok", "version": self._version}

                elif req_type == "GENERATE":
                    t0 = time.monotonic()
                    result = self._model.generate(
                        image=np.array(req["image"], dtype=np.uint8),
                        instruction=req["instruction"],
                        action_chunks_len=req.get("action_chunks_len", 8),
                        action_token_len=req.get("action_token_len", 7),
                        do_sample=req.get("do_sample", False),
                        temperature=req.get("temperature", 1.0),
                    )
                    logger.debug(f"GENERATE {time.monotonic()-t0:.2f}s")
                    resp = {
                        "type": "RESULT",
                        "request_id": req_id,
                        "status": "ok",
                        "action": result["action"],
                        "action_token_ids": result["action_token_ids"],
                        "input_token_ids": result["input_token_ids"],
                        "logprobs": result["logprobs"],
                        "version": self._version,
                    }

                elif req_type == "RELOAD_WEIGHTS":
                    ckpt = req.get("checkpoint_path", "")
                    if not os.path.exists(ckpt):
                        resp = {"type": "ACK", "request_id": req_id,
                                "status": "error",
                                "message": f"checkpoint not found: {ckpt}"}
                    else:
                        self._model.reload_weights(ckpt)
                        self._version = req.get("version", self._version + 1)
                        resp = {"type": "ACK", "request_id": req_id,
                                "status": "ok", "version": self._version}

                elif req_type == "SHUTDOWN":
                    resp = {"type": "ACK", "request_id": req_id, "status": "ok"}
                    self._send(client_id, resp)
                    self._running = False
                    break

                else:
                    resp = {"type": "ERROR", "request_id": req_id,
                            "message": f"unknown type: {req_type}"}

            except Exception as exc:
                logger.exception(f"Error handling {req_type}")
                resp = {"type": "ERROR", "request_id": req_id, "message": str(exc)}

            self._send(client_id, resp)

        self._socket.close()
        logger.info("Server stopped")

    def _send(self, client_id: bytes, response: dict) -> None:
        payload = pickle.dumps(response, protocol=4)
        self._socket.send_multipart([client_id, b"", payload])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(
        description="VLA Inference Server — run in the SimpleVLA conda env"
    )
    p.add_argument("--model_path", required=True,
                   help="HuggingFace model ID or local path")
    p.add_argument("--benchmark", default="libero_spatial",
                   help="LIBERO benchmark name used as unnorm_key "
                        "(e.g. libero_spatial, libero_10, libero_object)")
    p.add_argument("--address", default="tcp://*:5555",
                   help="ZMQ bind address (default: tcp://*:5555)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--no_center_crop", action="store_true")
    args = p.parse_args()

    model = OpenVLAOFTWrapper(
        model_path=args.model_path,
        unnorm_key=args.benchmark,
        device=args.device,
        center_crop=not args.no_center_crop,
    )
    server = VLAInferenceServer(model=model, address=args.address)
    server.run()


if __name__ == "__main__":
    main()
