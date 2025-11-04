"""Normalization utilities mirrored from the Wan diffusion stack.

The Wan VAE keeps a slimmed-down copy of RMSNorm / LayerNorm variants so it can
decode latents without importing the full diffusion model implementation.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (RMSNorm) used inside the Wan VAE stack.

    RMSNorm was introduced in *"Root Mean Square Layer Normalization"*
    by Zhang & Sennrich (2019) [arXiv:1910.07467], and later adopted by models
    such as T5 and Wan to improve numerical stability when training with
    reduced precision. Compared to LayerNorm, the mean is not subtracted,
    which keeps the variance normalization while avoiding the bias term.
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        """Initialize learnable scale parameters.

        Args:
            dim: Feature dimension to normalize (the `C` axis of `[B, L, C]` tensors).
            eps: Small constant added to the variance to avoid division by zero.
        """
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        """Apply RMS normalization with learned scaling.

        Args:
            x: Input tensor of shape `[B, L, C]` (batch, sequence/time, channel).

        Returns:
            Tensor: Normalized tensor with the same shape and dtype as `x`.
        """
        normed = self._norm(x.float()).type_as(x)
        return normed * self.weight.to(normed.dtype)

    def _norm(self, x: Tensor) -> Tensor:
        """Normalize by the root-mean-square value across the last dimension."""
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):
    """LayerNorm variant matching Wan defaults (float32 output, optional affine)."""

    def __init__(
        self, dim: int, eps: float = 1e-6, elementwise_affine: bool = False
    ) -> None:
        """Store LayerNorm hyper-parameters.

        Args:
            dim: Channel dimension of the input `[B, L, C]` tensor.
            eps: Small constant added to the variance for numerical stability.
            elementwise_affine: Whether to learn scale and bias parameters.
        """
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x: Tensor) -> Tensor:
        """Apply standard LayerNorm while preserving the input dtype.

        Args:
            x: Input tensor of shape `[B, L, C]`.

        Returns:
            Tensor: Layer-normalized representation with the same dtype/device.
        """
        return super().forward(x).type_as(x)
