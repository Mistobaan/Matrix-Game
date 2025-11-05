import torch

from torch import nn

from ..layers.convolution import CausalConv3d

from .residual import ResidualBlock
from .attention import AttentionBlock
from .transforms import Resample
from ..layers.normalization import RMS_norm

from .cache import FeatureCache


CACHE_T = 2


class Decoder3d(nn.Module):
    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_upsample=[False, True, True],
        dropout=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_upsample = temperal_upsample

        # dimensions
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        scale = 1.0 / 2 ** (len(dim_mult) - 2)

        # init block
        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(dims[0], dims[0], dropout),
            AttentionBlock(dims[0]),
            ResidualBlock(dims[0], dims[0], dropout),
        )

        # upsample blocks
        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            if i == 1 or i == 2 or i == 3:
                in_dim = in_dim // 2
            for _ in range(num_res_blocks + 1):
                upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    upsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            # upsample block
            if i != len(dim_mult) - 1:
                mode = "upsample3d" if temperal_upsample[i] else "upsample2d"
                upsamples.append(Resample(out_dim, mode=mode))
                scale *= 2.0
        self.upsamples = nn.Sequential(*upsamples)

        # output blocks
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False),
            nn.SiLU(),
            CausalConv3d(out_dim, 3, 3, padding=1),
        )

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        # conv1
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

        # middle
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        # upsamples
        for layer in self.upsamples:
            if feat_cache is not None:
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


class LatentInputProjector(nn.Module):
    """Prepare latent inputs for the decoder with a causal 1x1x1 projection."""

    def __init__(self, z_dim: int):
        super().__init__()
        self.proj = CausalConv3d(z_dim, z_dim, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.proj(z)


class TemporalDecoder(nn.Module):
    """Streaming decoder that mirrors the encoder's cache-aware workflow."""

    def __init__(
        self,
        decoder_module: nn.Module,
        latent_projector: LatentInputProjector,
        cache: FeatureCache,
    ):
        super().__init__()
        self.decoder_module = decoder_module
        self.latent_projector = latent_projector
        self.cache = cache

    def forward(self, z: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        if not use_cache:
            self.cache.clear()  # Ensure deterministic decoding when cache is unused
        latents = self.latent_projector(z)
        decoded_frames = []
        for frame_idx in range(latents.shape[2]):
            decoded_frames.append(
                self.decoder_module(
                    latents[:, :, frame_idx : frame_idx + 1, :, :],
                    feat_cache=self.cache.feature_map,
                    feat_idx=self.cache.reset_indices(),
                )
            )
        output = torch.cat(decoded_frames, dim=2)
        if not use_cache:
            self.cache.clear()  # Release cached activations after standalone decode
        return output
