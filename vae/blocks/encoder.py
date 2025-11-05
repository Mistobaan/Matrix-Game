import torch
from torch import nn

from typing import Tuple

from ..layers.convolution import CausalConv3d
from ..layers.normalization import RMS_norm

from .transforms import Resample
from .residual import ResidualBlock
from .cache import FeatureCache
from .attention import AttentionBlock


CACHE_T = 2


class LatentStatsProjector(nn.Module):
    """Thin wrapper that maps encoder features to mean/log-variance channels."""

    def __init__(self, z_dim: int):
        super().__init__()
        self.proj = CausalConv3d(z_dim * 2, z_dim * 2, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, log_var = self.proj(x).chunk(2, dim=1)
        return mu, log_var


class Encoder3d(nn.Module):
    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temporal_downsample=[True, True, False],
        dropout=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temporal_downsample = temporal_downsample

        # dimensions
        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0

        # init block
        self.conv1 = CausalConv3d(3, dims[0], 3, padding=1)

        # downsample blocks
        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            for _ in range(num_res_blocks):
                downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    downsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            # downsample block
            if i != len(dim_mult) - 1:
                mode = "downsample3d" if temporal_downsample[i] else "downsample2d"
                downsamples.append(Resample(out_dim, mode=mode))
                scale /= 2.0
        self.downsamples = nn.Sequential(*downsamples)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(out_dim, out_dim, dropout),
            AttentionBlock(out_dim),
            ResidualBlock(out_dim, out_dim, dropout),
        )

        # output blocks
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False),
            nn.SiLU(),
            CausalConv3d(out_dim, z_dim, 3, padding=1),
        )

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat(
                    [
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device),
                        cache_x,
                    ],
                    dim=2,
                )
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        # downsamples
        for layer in self.downsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        # middle
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        # head
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat(
                        [
                            feat_cache[idx][:, :, -1, :, :]
                            .unsqueeze(2)
                            .to(cache_x.device),
                            cache_x,
                        ],
                        dim=2,
                    )
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x


class TemporalEncoder(nn.Module):
    """Chunked encoder that streams frames while reusing cached activations."""

    def __init__(
        self,
        encoder_module: nn.Module,
        stats_projector: LatentStatsProjector,
        cache: FeatureCache,
        chunk_schedule: Tuple[int, int] = (1, 4),
    ):
        super().__init__()
        self.encoder_module = encoder_module
        self.stats_projector = stats_projector
        self.cache = cache
        self.first_length, self.block_length = chunk_schedule

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.cache.clear()  # Drop stale activations before streaming pass
        encoded_chunks = []
        for chunk in self._iter_chunks(x):
            encoded_chunks.append(
                self.encoder_module(
                    chunk,
                    feat_cache=self.cache.feature_map,
                    feat_idx=self.cache.reset_indices(),
                )
            )
        encoded = torch.cat(encoded_chunks, dim=2)
        mu, log_var = self.stats_projector(encoded)
        self.cache.clear()  # Reset cache so a fresh pass does not reuse state
        return mu, log_var

    def _iter_chunks(self, x: torch.Tensor):
        total_frames = x.shape[2]
        start = 0
        first_pass = True
        while start < total_frames:
            length = self.first_length if first_pass else self.block_length
            end = min(start + length, total_frames)
            yield x[:, :, start:end, :, :]
            start = end
            first_pass = False
