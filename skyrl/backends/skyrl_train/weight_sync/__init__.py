"""Weight synchronization abstractions for distributed RL training."""

import os
from typing import Type

from .base import LoraLoadRequest, WeightChunk, WeightUpdateRequest
from .broadcast_strategy import (
    BroadcastInitInfo,
    BroadcastTransferStrategy,
    BroadcastWeightTransferReceiver,
    BroadcastWeightTransferSender,
    BroadcastWeightUpdateRequest,
)
from .cuda_ipc_strategy import (
    CudaIpcInitInfo,
    CudaIpcTransferStrategy,
    CudaIpcWeightTransferReceiver,
    CudaIpcWeightTransferSender,
    CudaIpcWeightUpdateRequest,
)
from .transfer_strategy import (
    WeightSyncInitInfo,
    WeightTransferReceiver,
    WeightTransferSender,
    WeightTransferStrategy,
)
from .weight_extractor import WeightExtractor
from .weight_loader import WeightLoader


def get_transfer_strategy_cls(weight_sync_backend: str, colocate_all: bool) -> Type[WeightTransferStrategy]:
    """Get the appropriate transfer strategy class based on config.

    Uses CUDA IPC when:
    - weight_sync_backend is "nccl"
    - colocate_all is True (training and inference on same nodes)

    Otherwise uses broadcast.

    Args:
        weight_sync_backend: The weight sync backend ("nccl" or other).
        colocate_all: Whether training and inference are colocated on same nodes.

    Returns:
        The strategy class (CudaIpcTransferStrategy or BroadcastTransferStrategy).
    """
    strategy = get_transfer_strategy(weight_sync_backend, colocate_all)
    if strategy == "ipc":
        return CudaIpcTransferStrategy
    return BroadcastTransferStrategy


def get_transfer_strategy(weight_sync_backend: str, colocate_all: bool) -> str:
    """Get the appropriate transfer strategy string based on config."""
    force_broadcast = os.getenv("SKYRL_FORCE_BROADCAST_WEIGHT_SYNC", "").lower() in ("1", "true", "yes")
    force_broadcast = force_broadcast or os.getenv("SKYRL_FORCE_NCCL_WEIGHT_SYNC", "").lower() in (
        "1",
        "true",
        "yes",
    )
    if force_broadcast:
        return "nccl"
    if weight_sync_backend == "nccl" and colocate_all:
        return "ipc"
    return "nccl"


__all__ = [
    "WeightChunk",
    "WeightExtractor",
    "WeightLoader",
    "WeightUpdateRequest",
    "LoraLoadRequest",
    "BroadcastWeightUpdateRequest",
    "CudaIpcWeightUpdateRequest",
    "WeightTransferStrategy",
    "WeightTransferSender",
    "WeightTransferReceiver",
    "WeightSyncInitInfo",
    "BroadcastInitInfo",
    "CudaIpcInitInfo",
    "BroadcastTransferStrategy",
    "BroadcastWeightTransferSender",
    "BroadcastWeightTransferReceiver",
    "CudaIpcTransferStrategy",
    "CudaIpcWeightTransferSender",
    "CudaIpcWeightTransferReceiver",
    "get_transfer_strategy_cls",
]
