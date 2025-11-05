from typing import List, Optional
import torch


class FeatureCache:
    """Manage cached activations and indices for streaming 3D convolutions."""

    _len: int
    _feature_map: List[Optional[torch.Tensor]]
    _indices: List[int]

    def __init__(self, len: int):
        self._len = len
        self._feature_map = [None] * self._len
        self._indices = [0]

    def clear(self) -> None:
        """Reset cached tensors so subsequent passes start fresh."""
        self._feature_map = [None] * self._len
        self._indices = [0]

    @property
    def feature_map(self) -> List[Optional[torch.Tensor]]:
        return self._feature_map

    def reset_indices(self) -> List[int]:
        """Provide a fresh index list that downstream modules mutate in-place."""
        self._indices = [0]
        return self._indices
