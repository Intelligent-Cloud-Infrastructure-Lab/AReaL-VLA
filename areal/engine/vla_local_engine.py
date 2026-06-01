"""
VLALocalEngine — local inference engine for VLA models.

AReaL's default inference engines (RemoteSGLangEngine, RemoteVLLMEngine) require
the model to be served over HTTP.  VLA models such as OpenVLA-OFT (based on
Prismatic / PrismaticVLM) are not yet natively supported by SGLang / vLLM.

VLALocalEngine wraps a locally-loaded VLA model and exposes the same async
interface that VLARobotWorkflow expects (agenerate, get_version), so the workflow
code is unchanged regardless of whether the engine is remote or local.

Architecture note
-----------------
In AReaL, training and inference are separated: the actor (training engine) holds
the model weights and broadcasts updates to the inference engine.  With
VLALocalEngine, we bypass the HTTP inference server entirely — the model runs
inline on the same GPU(s) used for training.  This means:

  - No async training/inference overlap (the VLA model call blocks training).
  - Simpler setup: no SGLang server to launch or manage.
  - Suitable for single-node debugging and initial experiments.

For production scale, the recommended path is to add OpenVLA-OFT support to
SGLang and use AReaL's standard RemoteSGLangEngine.  The workflow code in
areal/workflow/vla_robot.py does not need to change when switching engines.

Weight update protocol
----------------------
AReaL's actor periodically calls actor.update_weights() which broadcasts new
parameters to the inference engine.  VLALocalEngine implements a simple
set_weights() method that copies updated parameters from the actor's state dict.
This mirrors the pattern in AReaL's FSDPPPOActor.connect_engine.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from typing import Any, Callable

import numpy as np
import torch

from areal.workflow.vla_robot import VLAEngine, VLAStepRequest, VLAStepResponse

logger = logging.getLogger("VLALocalEngine")


# ---------------------------------------------------------------------------
# Abstract base for any locally-hosted VLA model
# ---------------------------------------------------------------------------


class VLAModelBase:
    """
    Minimal interface that a VLA model must implement to work with VLALocalEngine.

    Implement this class (or adapt it) for your specific VLA model.
    See OpenVLAOFTModel below for a reference implementation targeting OpenVLA-OFT.
    """

    def generate_action_tokens(
        self,
        image: np.ndarray,
        instruction_ids: list[int],
        action_chunk_len: int,
        temperature: float = 1.0,
    ) -> tuple[list[int], list[float]]:
        """
        Run one VLA forward pass.

        Parameters
        ----------
        image:
            RGB observation from the robot camera, shape (H, W, 3), dtype uint8.
        instruction_ids:
            Tokenised language instruction (same across all steps of an episode).
        action_chunk_len:
            Number of action tokens to generate.
        temperature:
            Sampling temperature.  Use 0 for greedy decoding.

        Returns
        -------
        (action_token_ids, action_logprobs) where both lists have length
        action_chunk_len.
        """
        raise NotImplementedError

    def get_image_token_ids(self, image: np.ndarray) -> list[int]:
        """
        Encode an RGB image to discrete image token IDs.

        Only needed for models that encode images into the token vocabulary.
        Models using a continuous vision encoder (ViT) can return [] here.
        """
        return []

    def get_input_token_ids(
        self, image: np.ndarray, instruction_ids: list[int]
    ) -> list[int]:
        """
        Return the full prefix token sequence that will appear in input_ids.

        This is used to build the trajectory's input_ids tensor.  Typically
        equals image_token_ids + instruction_ids (in model-specific order).
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# VLALocalEngine
# ---------------------------------------------------------------------------


class VLALocalEngine(VLAEngine):
    """
    Inference engine that wraps a locally-loaded VLA model.

    Parameters
    ----------
    model:
        Instance of VLAModelBase (or a compatible duck-typed class).
    action_chunk_len:
        Number of action tokens per generation call.
    temperature:
        Sampling temperature for action token generation.
    max_workers:
        Size of the ThreadPoolExecutor used to run blocking model forward passes
        without blocking the asyncio event loop.
    """

    def __init__(
        self,
        model: VLAModelBase,
        action_chunk_len: int = 7,
        temperature: float = 1.0,
        max_workers: int = 1,
    ) -> None:
        self.model = model
        self.action_chunk_len = action_chunk_len
        self.temperature = temperature
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="vla_local"
        )
        self._version: int = 0

    # ------------------------------------------------------------------
    # VLAEngine interface
    # ------------------------------------------------------------------

    async def agenerate(self, req: VLAStepRequest) -> VLAStepResponse:
        """
        Async wrapper around the VLA model's blocking forward pass.

        The ThreadPoolExecutor prevents the model call from stalling the
        asyncio event loop so that multiple robot environments can run
        concurrently (each awaiting its own agenerate call).
        """
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            self._executor,
            self._sync_generate,
            req,
        )
        return response

    def _sync_generate(self, req: VLAStepRequest) -> VLAStepResponse:
        """Synchronous VLA forward pass (runs in thread pool)."""
        with torch.no_grad():
            action_token_ids, action_logprobs = self.model.generate_action_tokens(
                image=req.image,
                instruction_ids=req.instruction_ids,
                action_chunk_len=req.max_new_tokens,
                temperature=self.temperature,
            )
            input_token_ids = self.model.get_input_token_ids(
                image=req.image,
                instruction_ids=req.instruction_ids,
            )

        return VLAStepResponse(
            input_tokens=input_token_ids,
            output_tokens=action_token_ids,
            output_logprobs=action_logprobs,
            output_versions=[self._version] * len(action_token_ids),
        )

    def get_version(self) -> int:
        return self._version

    def set_version(self, version: int) -> None:
        self._version = version

    # ------------------------------------------------------------------
    # Weight update (called by actor.update_weights via connect_engine)
    # ------------------------------------------------------------------

    def set_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        """
        Load updated model weights into the local VLA model.

        AReaL's actor calls this after each gradient update step to keep the
        inference model in sync with the training model.
        """
        # Filter out keys not in the VLA model (e.g. optimizer state)
        model_state = self.model.state_dict() if hasattr(self.model, "state_dict") else {}
        compatible = {k: v for k, v in state_dict.items() if k in model_state}
        if hasattr(self.model, "load_state_dict"):
            self.model.load_state_dict(compatible, strict=False)
        logger.debug(
            f"[VLALocalEngine] weight update loaded {len(compatible)}/{len(state_dict)} keys"
        )

    def destroy(self) -> None:
        """Clean up the thread pool executor."""
        self._executor.shutdown(wait=False)


# ---------------------------------------------------------------------------
# Reference: OpenVLA-OFT model adapter
# ---------------------------------------------------------------------------


class OpenVLAOFTModel(VLAModelBase):
    """
    Adapter for OpenVLA-OFT (Open Vision-Language-Action, OFT variant).

    OpenVLA-OFT discretises the action space into 256 bins per dimension and
    autoregressively generates action_dim × chunk_len tokens.  The model is
    based on PrismaticVLM (LLaVA backbone + SigLIP vision encoder).

    This adapter handles:
      - Action token vocabulary mapping (token IDs in [32000, 32256])
      - Bin decoding back to continuous actions (uniform grid in [-1, 1])
      - Prompt construction (image special token + instruction text)

    Usage
    -----
        from transformers import AutoProcessor
        model = AutoModelForVision2Seq.from_pretrained(...)
        processor = AutoProcessor.from_pretrained(...)
        vla_model = OpenVLAOFTModel(model, processor, action_chunk_len=7)
        engine = VLALocalEngine(vla_model, action_chunk_len=7)
    """

    # OpenVLA-OFT discretises each action dimension into N_BINS = 256 bins
    N_BINS: int = 256
    # Token IDs for action bins in OpenVLA-OFT's vocabulary
    ACTION_TOKEN_BEGIN: int = 32000
    ACTION_TOKEN_END: int = 32256  # exclusive

    def __init__(
        self,
        model: Any,  # transformers.AutoModelForVision2Seq
        processor: Any,  # transformers.AutoProcessor
        action_chunk_len: int = 7,
        action_dim: int = 7,
        action_scale: float = 1.0,
        device: str | torch.device = "cuda",
    ) -> None:
        self.model = model
        self.processor = processor
        self.action_chunk_len = action_chunk_len
        self.action_dim = action_dim
        self.action_scale = action_scale
        self.device = torch.device(device)

        # Move model to device (if not already)
        if hasattr(model, "to"):
            self.model = model.to(self.device)

    # ------------------------------------------------------------------
    # VLAModelBase interface
    # ------------------------------------------------------------------

    def generate_action_tokens(
        self,
        image: np.ndarray,
        instruction_ids: list[int],
        action_chunk_len: int,
        temperature: float = 1.0,
    ) -> tuple[list[int], list[float]]:
        """
        Run OpenVLA-OFT forward pass and return action token IDs + logprobs.

        The model generates action_dim × action_chunk_len tokens total
        (= total_tokens = 7 × 7 = 49 for the default configuration).
        """
        from PIL import Image as PILImage  # type: ignore[import]
        import torch.nn.functional as F

        pil_image = PILImage.fromarray(image)
        n_tokens = action_dim = self.action_dim
        total_tokens = action_chunk_len * action_dim

        # Build prompt
        prompt = self._build_prompt(instruction_ids)

        # Tokenise (processor handles image + text jointly)
        inputs = self.processor(
            text=prompt,
            images=pil_image,
            return_tensors="pt",
        ).to(self.device)

        # Greedy or sampled generation
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=total_tokens,
                do_sample=(temperature > 0),
                temperature=temperature if temperature > 0 else None,
                output_scores=True,
                return_dict_in_generate=True,
            )

        generated_ids: list[int] = (
            output.sequences[0, inputs["input_ids"].shape[-1]:].tolist()
        )[:total_tokens]

        # Compute per-token logprobs from scores
        logprobs: list[float] = []
        for step_idx, scores in enumerate(output.scores[:total_tokens]):
            probs = F.softmax(scores[0], dim=-1)
            tok_id = generated_ids[step_idx] if step_idx < len(generated_ids) else 0
            lp = float(torch.log(probs[tok_id] + 1e-10).item())
            logprobs.append(lp)

        # Pad if generation stopped early
        while len(generated_ids) < total_tokens:
            generated_ids.append(self.ACTION_TOKEN_BEGIN)
            logprobs.append(-100.0)

        return generated_ids, logprobs

    def get_input_token_ids(
        self, image: np.ndarray, instruction_ids: list[int]
    ) -> list[int]:
        """
        Return the prompt token IDs (image tokens + instruction tokens).

        In OpenVLA-OFT, the prompt consists of a special [IMG] placeholder token
        expanded by the vision encoder, followed by the tokenised instruction.
        We return the full token sequence that the model actually processes.
        """
        from PIL import Image as PILImage  # type: ignore[import]

        pil_image = PILImage.fromarray(image)
        prompt = self._build_prompt(instruction_ids)
        inputs = self.processor(
            text=prompt,
            images=pil_image,
            return_tensors="pt",
        )
        return inputs["input_ids"][0].tolist()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_prompt(self, instruction_ids: list[int]) -> str:
        """Decode instruction IDs back to a string for the processor."""
        # The processor re-tokenises; we decode + re-encode to stay consistent
        text = self.processor.tokenizer.decode(
            instruction_ids, skip_special_tokens=True
        )
        # OpenVLA-OFT prompt format:
        # "In: What action should the robot take to <instruction>?\nOut:"
        return f"In: What action should the robot take to {text}?\nOut:"


# ---------------------------------------------------------------------------
# Action decoder: token IDs → continuous robot action
# ---------------------------------------------------------------------------


def make_openvla_action_decoder(
    action_dim: int = 7,
    action_chunk_len: int = 7,
    n_bins: int = 256,
    action_min: float = -1.0,
    action_max: float = 1.0,
) -> Callable[[list[int]], np.ndarray]:
    """
    Build a closure that decodes OpenVLA-OFT action token IDs to a numpy array.

    Each action token encodes one scalar action value as a bin index in
    [ACTION_TOKEN_BEGIN, ACTION_TOKEN_BEGIN + n_bins).  The decoding formula:

        bin_idx = token_id - ACTION_TOKEN_BEGIN
        value   = action_min + (bin_idx + 0.5) * (action_max - action_min) / n_bins

    The output shape is (action_chunk_len, action_dim).

    Parameters
    ----------
    action_dim:
        Number of degrees of freedom (7 for OpenVLA-OFT on LIBERO / RoboTwin).
    action_chunk_len:
        Number of action steps per VLA generation call.
    n_bins:
        Discretisation bins per action dimension.
    action_min / action_max:
        Range of the continuous action space.
    """
    token_begin = OpenVLAOFTModel.ACTION_TOKEN_BEGIN
    bin_width = (action_max - action_min) / n_bins

    def decode(token_ids: list[int]) -> np.ndarray:
        if len(token_ids) != action_dim * action_chunk_len:
            # Pad or truncate gracefully
            token_ids = (token_ids + [token_begin] * action_dim * action_chunk_len)[
                : action_dim * action_chunk_len
            ]
        actions = np.zeros((action_chunk_len, action_dim), dtype=np.float32)
        for step in range(action_chunk_len):
            for dim in range(action_dim):
                idx = step * action_dim + dim
                bin_idx = token_ids[idx] - token_begin
                bin_idx = int(np.clip(bin_idx, 0, n_bins - 1))
                actions[step, dim] = action_min + (bin_idx + 0.5) * bin_width
        return actions

    return decode


# Callable type alias used in type hints elsewhere in the codebase
Callable = type(lambda: None)
