"""
VLA-specific config extension of GRPOConfig.

Kept in a separate importable module (not __main__) so AReaL's worker
processes can deserialize the config correctly via RPCSerialization.
"""
from dataclasses import dataclass
from areal.api.cli_args import GRPOConfig


@dataclass
class VLAGRPOConfig(GRPOConfig):
    """GRPOConfig extended with VLA robot training fields."""
    benchmark: str = "libero_spatial"
    n_seeds_per_task: int = 5
    val_fraction: float = 0.1
    inference_server_address: str = "tcp://localhost:5556"
    action_chunks_len: int = 8
    max_episode_steps: int = 512
    unnorm_key: str = "libero_spatial_no_noops"
