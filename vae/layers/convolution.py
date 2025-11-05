from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class CausalConv3d(nn.Conv3d):
    """3D convolution layer that enforces temporal causality.

    The layer mirrors ``nn.Conv3d`` but guarantees that no future temporal
    information leaks into the current frame by padding exclusively on the
    past. It is primarily used by streaming pipelines that process video
    sequences incrementally.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._padding: Tuple[int, int, int, int, int, int] = (
            self.padding[2],
            self.padding[2],
            self.padding[1],
            self.padding[1],
            2 * self.padding[0],
            0,
        )
        self.padding = (0, 0, 0)

    def forward(
        self, x: torch.Tensor, cache_x: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply causal convolution over the input tensor.

        Args:
            x: Input tensor of shape ``[batch, channels, frames, height, width]``.
            cache_x: Optional tensor containing cached frames along the temporal
                axis taken from the previous invocation. This enables streaming
                inference by prepending prior context instead of reprocessing
                the entire sequence.

        Returns:
            Convolved tensor containing only information from the current and
            past frames.

        Raises:
            ValueError: If ``cache_x`` has an incompatible temporal dimension.
        """
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            if cache_x.shape[2] > self._padding[4]:
                raise ValueError(
                    "cache_x temporal length exceeds the configured causal padding."
                )
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)

        return super().forward(x)
