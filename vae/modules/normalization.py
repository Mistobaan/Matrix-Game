import torch
from torch import nn
from torch import Tensor


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
