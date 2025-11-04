# High-level overview of how the Matrix-Game inference stack stitches Wan VAE
# encoding/decoding together with the causal Wan diffusion backbone.

#           +-------------------------+        +----------------------------+
#           |  WanVAEWrapper          |        |  WanDiffusionWrapper       |
#           |-------------------------|        |----------------------------|
#           | - load Wan 2.1 VAE      |        | - load CausalWanModel      |
#  Reference| - encode RGB frames --->|--------> - fuse history + cond      |
#   Frames  | - decode latent output  |        | - FlowMatch scheduling     |
#           +-------------------------+        | - KV-cached causal blocks  |
#                                              +-------------+--------------+
#                                                            |
#                                              +-------------v--------------+
#                                              |       CausalWanModel       |
#                                              |----------------------------|
#                                              | Patch embed 3D latents     |
#                                              | Time/cond projection       |
#                                              | Blockwise causal attention |
#                                              | Action conditioning hooks  |
#                                              | Linear head + unpatchify   |
#                                              +----------------------------+

# Key data flow:
#     1. `WanVAEWrapper` normalizes input frames, producing latent tensors.
#     2. `WanDiffusionWrapper` packages latents, scheduler, and conditioning.
#     3. `CausalWanModel` performs causal denoising with action-aware modules.
#     4. Output latents are decoded back to pixels via `WanVAEWrapper`.

import types
from typing import List, Optional, Tuple
import torch
from torch import nn
from einops import rearrange
from utils.scheduler import SchedulerInterface, FlowMatchScheduler
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.modeling_utils import ModelMixin
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention
from wan.modules.attention import attention
from wan.modules.model import (
    WAN_CROSSATTENTION_CLASSES,
    MLPProj,
    WanLayerNorm,
    WanRMSNorm,
    rope_apply,
    rope_params,
    sinusoidal_embedding_1d,
)
from wan.modules.action_module import ActionModule
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

__all__ = [
    'WanVAE',
    'CausalWanModel',
]

CACHE_T = 2


def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    f, h, w = grid_sizes.tolist()

    for i in range(len(x)):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).type_as(x)


class CausalWanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.max_attention_size = 15 * 1 * 880 if local_attn_size == -1 else local_attn_size * 880
        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(
        self,
        x,
        seq_lens,
        grid_sizes,
        freqs,
        block_mask,
        kv_cache=None,
        current_start=0,
        cache_start=None
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C] # num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            block_mask (BlockMask)
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        if cache_start is None:
            cache_start = current_start

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)  # B, F, HW, C

        if kv_cache is None:
            roped_query = rope_apply(q, grid_sizes, freqs).type_as(v)
            roped_key = rope_apply(k, grid_sizes, freqs).type_as(v)

            padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
            padded_roped_query = torch.cat(
                [roped_query,
                 torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                             device=q.device, dtype=v.dtype)],
                dim=1
            )

            padded_roped_key = torch.cat(
                [roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                        device=k.device, dtype=v.dtype)],
                dim=1
            )

            padded_v = torch.cat(
                [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                device=v.device, dtype=v.dtype)],
                dim=1
            )

            x = flex_attention(
                query=padded_roped_query.transpose(2, 1),  # after: B, HW, F, C
                key=padded_roped_key.transpose(2, 1),
                value=padded_v.transpose(2, 1),
                block_mask=block_mask
            )[:, :, :-padded_length].transpose(2, 1)
        else:
            assert grid_sizes.ndim == 1
            frame_seqlen = math.prod(grid_sizes[1:]).item()
            current_start_frame = current_start // frame_seqlen
            roped_query = causal_rope_apply(
                q, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)
            roped_key = causal_rope_apply(
                k, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)

            current_end = current_start + roped_query.shape[1]
            sink_tokens = self.sink_size * frame_seqlen

            kv_cache_size = kv_cache["k"].shape[1]
            num_new_tokens = roped_query.shape[1]

            if (current_end > kv_cache["global_end_index"].item()) and (
                    num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size):

                num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
                num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens
                kv_cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    kv_cache["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                kv_cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    kv_cache["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                # Insert the new keys/values at the end
                local_end_index = kv_cache["local_end_index"].item() + current_end - \
                    kv_cache["global_end_index"].item() - num_evicted_tokens
                local_start_index = local_end_index - num_new_tokens
                kv_cache["k"][:, local_start_index:local_end_index] = roped_key
                kv_cache["v"][:, local_start_index:local_end_index] = v
            else:
                # Assign new keys/values directly up to current_end
                local_end_index = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item()
                local_start_index = local_end_index - num_new_tokens

                kv_cache["k"][:, local_start_index:local_end_index] = roped_key
                kv_cache["v"][:, local_start_index:local_end_index] = v
            x = attention(
                roped_query,
                kv_cache["k"][:, max(0, local_end_index - self.max_attention_size):local_end_index],
                kv_cache["v"][:, max(0, local_end_index - self.max_attention_size):local_end_index]
            )
            kv_cache["global_end_index"].fill_(current_end)
            kv_cache["local_end_index"].fill_(local_end_index)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class CausalWanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 cross_attn_norm=False,
                 action_config={},
                 block_idx=0,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        if len(action_config) != 0 and block_idx in action_config['blocks']:
            self.action_model = ActionModule(**action_config, local_attn_size=self.local_attn_size)
        else:
            self.action_model = None
        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(dim, num_heads, local_attn_size, sink_size, qk_norm, eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        block_mask,
        block_mask_mouse,
        block_mask_keyboard,
        num_frame_per_block=3,
        use_rope_keyboard=False,
        mouse_cond=None,
        keyboard_cond=None,
        kv_cache=None,
        kv_cache_mouse=None,
        kv_cache_keyboard=None,
        crossattn_cache=None,
        current_start=0,
        cache_start=None,
        context_lens=None
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, F, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        assert e.ndim == 4
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]

        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)

        y = self.self_attn(
            (self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]).flatten(1, 2),
            seq_lens, grid_sizes,
            freqs, block_mask, kv_cache, current_start, cache_start)

        x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[2]).flatten(1, 2)

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, e, mouse_cond, keyboard_cond, block_mask_mouse, block_mask_keyboard, kv_cache_mouse=None, kv_cache_keyboard=None, crossattn_cache=None, start_frame=0, use_rope_keyboard=False, num_frame_per_block=3):
            x = x + self.cross_attn(self.norm3(x.to(context.dtype)), context, crossattn_cache=crossattn_cache)
            if self.action_model is not None:
                assert mouse_cond is not None or keyboard_cond is not None
                x = self.action_model(x.to(context.dtype), grid_sizes[0], grid_sizes[1], grid_sizes[2], mouse_cond, keyboard_cond, block_mask_mouse, block_mask_keyboard, is_causal=True, kv_cache_mouse=kv_cache_mouse, kv_cache_keyboard=kv_cache_keyboard, start_frame=start_frame, use_rope_keyboard=use_rope_keyboard, num_frame_per_block=num_frame_per_block)

            y = self.ffn(
                (self.norm2(x).unflatten(dim=1, sizes=(num_frames,
                 frame_seqlen)) * (1 + e[4]) + e[3]).flatten(1, 2)
            )

            x = x + (y.unflatten(dim=1, sizes=(num_frames,
                     frame_seqlen)) * e[5]).flatten(1, 2)
            return x
        assert grid_sizes.ndim == 1
        x = cross_attn_ffn(x, context, e, mouse_cond, keyboard_cond, block_mask_mouse, block_mask_keyboard, kv_cache_mouse, kv_cache_keyboard, crossattn_cache, start_frame=current_start // math.prod(grid_sizes[1:]).item(), use_rope_keyboard=use_rope_keyboard, num_frame_per_block=num_frame_per_block)
        return x


class CausalHead(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, F, 1, C]
        """

        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        x = (self.head(self.norm(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]))
        return x


class CausalWanModel(ModelMixin, ConfigMixin, FromOriginalModelMixin, PeftAdapterMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim'
    ]
    _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=36,
                 dim=1536,
                 ffn_dim=8960,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=12,
                 num_layers=30,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 cross_attn_norm=True,
                 action_config={},
                 eps=1e-6):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            local_attn_size (`int`, *optional*, defaults to -1):
                Window size for temporal local attention (-1 indicates global attention)
            sink_size (`int`, *optional*, defaults to 0):
                Size of the attention sink, we keep the first `sink_size` frames unchanged when rolling the KV cache
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['i2v']
        self.model_type = model_type
        self.use_action_module = len(action_config) > 0
        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            CausalWanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                                    local_attn_size, sink_size, qk_norm, cross_attn_norm, action_config=action_config, eps=eps, block_idx=idx)
            for idx in range(num_layers)
        ])

        # head
        self.head = CausalHead(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
            dim=1)

        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)

        # initialize weights
        self.init_weights()

        self.gradient_checkpointing = False

        self.block_mask = None
        self.block_mask_keyboard = None
        self.block_mask_mouse = None
        self.use_rope_keyboard = True
        self.num_frame_per_block = 1

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    @staticmethod
    def _prepare_blockwise_causal_attn_mask(
        device: torch.device | str, num_frames: int = 9,
        frame_seqlen: int = 880, num_frame_per_block=1, local_attn_size=-1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=0,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for tmp in frame_indices:
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | (q_idx == kv_idx)

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        import torch.distributed as dist
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames")

        return block_mask

    @staticmethod
    def _prepare_blockwise_causal_attn_mask_keyboard(
        device: torch.device | str, num_frames: int = 9,
        frame_seqlen: int = 880, num_frame_per_block=1, local_attn_size=-1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        total_length2 = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length2 = math.ceil(total_length2 / 32) * 32 - total_length2
        padded_length_kv2 = math.ceil(num_frames / 32) * 32 - num_frames
        ends2 = torch.zeros(total_length2 + padded_length2,
                           device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices2 = torch.arange(
            start=0,
            end=total_length2,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )
        cnt = num_frame_per_block
        for tmp in frame_indices2:
            ends2[tmp:tmp + frame_seqlen * num_frame_per_block] = cnt
            cnt += num_frame_per_block

        def attention_mask2(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends2[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends2[q_idx]) & (kv_idx >= (ends2[q_idx] - local_attn_size))) | (q_idx == kv_idx)

        block_mask2 = create_block_mask(attention_mask2, B=None, H=None, Q_LEN=total_length2 + padded_length2,
                                       KV_LEN=num_frames + padded_length_kv2, _compile=False, device=device)

        import torch.distributed as dist
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames")

        return block_mask2

    @staticmethod
    def _prepare_blockwise_causal_attn_mask_action(
        device: torch.device | str, num_frames: int = 9,
        frame_seqlen: int = 1, num_frame_per_block=1, local_attn_size=-1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        total_length2 = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length2 = math.ceil(total_length2 / 32) * 32 - total_length2
        padded_length_kv2 = math.ceil(num_frames / 32) * 32 - num_frames
        ends2 = torch.zeros(total_length2 + padded_length2,
                           device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices2 = torch.arange(
            start=0,
            end=total_length2,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )
        cnt = num_frame_per_block
        for tmp in frame_indices2:
            ends2[tmp:tmp + frame_seqlen * num_frame_per_block] = cnt
            cnt += num_frame_per_block

        def attention_mask2(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends2[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends2[q_idx]) & (kv_idx >= (ends2[q_idx] - local_attn_size))) | (q_idx == kv_idx)

        block_mask2 = create_block_mask(attention_mask2, B=None, H=None, Q_LEN=total_length2 + padded_length2,
                                       KV_LEN=num_frames + padded_length_kv2, _compile=False, device=device)

        import torch.distributed as dist
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames")

        return block_mask2

    def _forward_inference(
        self,
        x,
        t,
        visual_context, cond_concat, mouse_cond=None, keyboard_cond=None,
        kv_cache: dict = None,
        kv_cache_mouse=None,
        kv_cache_keyboard=None,
        crossattn_cache: dict = None,
        current_start: int = 0,
        cache_start: int = 0
    ):
        r"""
        Run the diffusion model with kv caching.
        See Algorithm 2 of CausVid paper https://arxiv.org/abs/2412.07772 for details.
        This function will be run for num_frame times.
        Process the latent frames one by one (1560 tokens each)

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """

        if mouse_cond is not None or keyboard_cond is not None:
            assert self.use_action_module is True
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        x = torch.cat([x, cond_concat], dim=1)  # B C' F H W

        # embeddings
        x = self.patch_embedding(x)
        grid_sizes = torch.tensor(x.shape[2:], dtype=torch.long)
        x = x.flatten(2).transpose(1, 2)  # B FHW C'
        seq_lens = torch.tensor([u.size(0) for u in x], dtype=torch.long)
        assert seq_lens[0] <= 15 * 1 * 880

        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
        # context
        context_lens = None
        context = self.img_emb(visual_context)
        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            mouse_cond=mouse_cond,
            context_lens=context_lens,
            keyboard_cond=keyboard_cond,
            block_mask=self.block_mask,
            block_mask_mouse=self.block_mask_mouse,
            block_mask_keyboard=self.block_mask_keyboard,
            use_rope_keyboard=self.use_rope_keyboard,
            num_frame_per_block=self.num_frame_per_block
        )

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        for block_index, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "kv_cache_mouse": kv_cache_mouse[block_index],
                        "kv_cache_keyboard": kv_cache_keyboard[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start,
                    }

                )
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "kv_cache_mouse": kv_cache_mouse[block_index],
                        "kv_cache_keyboard": kv_cache_keyboard[block_index],
                        "crossattn_cache": crossattn_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start,
                    }
                )
                x = block(x, **kwargs)

        # head
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return x

    def _forward_train(
        self,
        x,
        t,
        visual_context, cond_concat, mouse_cond=None, keyboard_cond=None,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        # params
        if mouse_cond is not None or keyboard_cond is not None:
            assert self.use_action_module is True
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)
        x = torch.cat([x, cond_concat], dim=1)
        # Construct blockwise causal attn mask
        if self.block_mask is None:
            self.block_mask = self._prepare_blockwise_causal_attn_mask(
                device, num_frames=x.shape[2],
                frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                num_frame_per_block=self.num_frame_per_block,
                local_attn_size=self.local_attn_size
            )
        if self.block_mask_keyboard is None:
            if self.use_rope_keyboard is False:
                self.block_mask_keyboard = self._prepare_blockwise_causal_attn_mask_keyboard(
                    device, num_frames=x.shape[2],
                    frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                    num_frame_per_block=self.num_frame_per_block,
                    local_attn_size=self.local_attn_size
                )
            else:
                self.block_mask_keyboard = self._prepare_blockwise_causal_attn_mask_action(
                    device, num_frames=x.shape[2],
                    frame_seqlen=1,
                    num_frame_per_block=self.num_frame_per_block,
                    local_attn_size=self.local_attn_size
            )
        if self.block_mask_mouse is None:
            self.block_mask_mouse = self._prepare_blockwise_causal_attn_mask_action(
                device, num_frames=x.shape[2],
                frame_seqlen=1,
                num_frame_per_block=self.num_frame_per_block,
                local_attn_size=self.local_attn_size
            )
        x = self.patch_embedding(x)
        grid_sizes = torch.tensor(x.shape[2:], dtype=torch.long)
        x = x.flatten(2).transpose(1, 2)
        seq_lens = torch.tensor([u.size(0) for u in x], dtype=torch.long)
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)

        context_lens = None
        context = self.img_emb(visual_context)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            mouse_cond=mouse_cond,
            context_lens=context_lens,
            keyboard_cond=keyboard_cond,
            block_mask=self.block_mask,
            block_mask_mouse=self.block_mask_mouse,
            block_mask_keyboard=self.block_mask_keyboard,
            use_rope_keyboard=self.use_rope_keyboard,
            num_frame_per_block=self.num_frame_per_block
        )

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                x = block(x, **kwargs)

        # head
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return x

    def forward(
        self,
        *args,
        **kwargs
    ):
        if kwargs.get('kv_cache', None) is not None:
            return self._forward_inference(*args, **kwargs)
        else:
            return self._forward_train(*args, **kwargs)

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        bs = x.shape[0]
        x = x.view(bs, *grid_sizes, *self.patch_size, c)
        x = torch.einsum("bfhwpqrc->bcfphqwr", x)
        x = x.reshape(bs, c, *[i * j for i, j in zip(grid_sizes, self.patch_size)])
        return x

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))

        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
        if self.use_action_module is True:
            for m in self.blocks:
                try:
                    nn.init.zeros_(m.action_model.proj_mouse.weight)
                    if m.action_model.proj_mouse.bias is not None:
                        nn.init.zeros_(m.action_model.proj_mouse.bias)
                    nn.init.zeros_(m.action_model.proj_keyboard.weight)
                    if m.action_model.proj_keyboard.bias is not None:
                        nn.init.zeros_(m.action_model.proj_keyboard.bias)
                except Exception:
                    pass


class CausalConv3d(nn.Conv3d):
    """
    Causal 3d convolution.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._padding = (self.padding[2], self.padding[2], self.padding[1],
                         self.padding[1], 2 * self.padding[0], 0)
        self.padding = (0, 0, 0)

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)

        return super().forward(x)


class RMS_norm(nn.Module):

    def __init__(self, dim, channel_first=True, images=True, bias=False):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.

    def forward(self, x):
        return F.normalize(
            x, dim=(1 if self.channel_first else
                    -1)) * self.scale * self.gamma + self.bias


class Upsample(nn.Upsample):

    def forward(self, x):
        """
        Fix bfloat16 support for nearest neighbor interpolation.
        """
        return super().forward(x.float()).type_as(x)


class Resample(nn.Module):

    def __init__(self, dim, mode):
        assert mode in ('none', 'upsample2d', 'upsample3d', 'downsample2d',
                        'downsample3d')
        super().__init__()
        self.dim = dim
        self.mode = mode

        # layers
        if mode == 'upsample2d':
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2., 2.), mode='nearest'),
                nn.Conv2d(dim, dim // 2, 3, padding=1))
        elif mode == 'upsample3d':
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2., 2.), mode='nearest'),
                nn.Conv2d(dim, dim // 2, 3, padding=1))
            self.time_conv = CausalConv3d(
                dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))

        elif mode == 'downsample2d':
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)))
        elif mode == 'downsample3d':
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)))
            self.time_conv = CausalConv3d(
                dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))

        else:
            self.resample = nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        b, c, t, h, w = x.size()
        if self.mode == 'upsample3d':
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = 'Rep'
                    feat_idx[0] += 1
                else:

                    cache_x = x[:, :, -CACHE_T:, :, :].clone()
                    if cache_x.shape[2] < 2 and feat_cache[
                            idx] is not None and feat_cache[idx] != 'Rep':
                        # cache last frame of last two chunk
                        cache_x = torch.cat([
                            feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                                cache_x.device), cache_x
                        ],
                            dim=2)
                    if cache_x.shape[2] < 2 and feat_cache[
                            idx] is not None and feat_cache[idx] == 'Rep':
                        cache_x = torch.cat([
                            torch.zeros_like(cache_x).to(cache_x.device),
                            cache_x
                        ],
                            dim=2)
                    if feat_cache[idx] == 'Rep':
                        x = self.time_conv(x)
                    else:
                        x = self.time_conv(x, feat_cache[idx])
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1

                    x = x.reshape(b, 2, c, t, h, w)
                    x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]),
                                    3)
                    x = x.reshape(b, c, t * 2, h, w)
        t = x.shape[2]
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.resample(x)
        x = rearrange(x, '(b t) c h w -> b c t h w', t=t)

        if self.mode == 'downsample3d':
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x.clone()
                    feat_idx[0] += 1
                else:

                    cache_x = x[:, :, -1:, :, :].clone()
                    # if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx]!='Rep':
                    #     # cache last frame of last two chunk
                    #     cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)

                    x = self.time_conv(
                        torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
        return x

    def init_weight(self, conv):
        conv_weight = conv.weight
        nn.init.zeros_(conv_weight)
        c1, c2, t, h, w = conv_weight.size()
        one_matrix = torch.eye(c1, c2)
        init_matrix = one_matrix
        nn.init.zeros_(conv_weight)
        # conv_weight.data[:,:,-1,1,1] = init_matrix * 0.5
        conv_weight.data[:, :, 1, 0, 0] = init_matrix  # * 0.5
        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def init_weight2(self, conv):
        conv_weight = conv.weight.data
        nn.init.zeros_(conv_weight)
        c1, c2, t, h, w = conv_weight.size()
        init_matrix = torch.eye(c1 // 2, c2)
        # init_matrix = repeat(init_matrix, 'o ... -> (o 2) ...').permute(1,0,2).contiguous().reshape(c1,c2)
        conv_weight[:c1 // 2, :, -1, 0, 0] = init_matrix
        conv_weight[c1 // 2:, :, -1, 0, 0] = init_matrix
        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)


class ResidualBlock(nn.Module):

    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # layers
        self.residual = nn.Sequential(
            RMS_norm(in_dim, images=False), nn.SiLU(),
            CausalConv3d(in_dim, out_dim, 3, padding=1),
            RMS_norm(out_dim, images=False), nn.SiLU(), nn.Dropout(dropout),
            CausalConv3d(out_dim, out_dim, 3, padding=1))
        self.shortcut = CausalConv3d(in_dim, out_dim, 1) \
            if in_dim != out_dim else nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        h = self.shortcut(x)
        for layer in self.residual:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                            cache_x.device), cache_x
                    ],
                        dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x + h


class AttentionBlock(nn.Module):
    """
    Causal self-attention with a single head.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # layers
        self.norm = RMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

        # zero out the last layer params
        nn.init.zeros_(self.proj.weight)

    def forward(self, x):
        identity = x
        b, c, t, h, w = x.size()
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.norm(x)
        # compute query, key, value
        q, k, v = self.to_qkv(x).reshape(b * t, 1, c * 3,
                                         -1).permute(0, 1, 3,
                                                     2).contiguous().chunk(
                                                         3, dim=-1)

        # apply attention
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
        )
        x = x.squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w)

        # output
        x = self.proj(x)
        x = rearrange(x, '(b t) c h w-> b c t h w', t=t)
        return x + identity


class Encoder3d(nn.Module):

    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_downsample=[True, True, False],
                 dropout=0.0):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample

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
                mode = 'downsample3d' if temperal_downsample[
                    i] else 'downsample2d'
                downsamples.append(Resample(out_dim, mode=mode))
                scale /= 2.0
        self.downsamples = nn.Sequential(*downsamples)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(out_dim, out_dim, dropout), AttentionBlock(out_dim),
            ResidualBlock(out_dim, out_dim, dropout))

        # output blocks
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False), nn.SiLU(),
            CausalConv3d(out_dim, z_dim, 3, padding=1))

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([
                    feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                        cache_x.device), cache_x
                ],
                    dim=2)
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
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                            cache_x.device), cache_x
                    ],
                        dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x


class Decoder3d(nn.Module):

    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_upsample=[False, True, True],
                 dropout=0.0):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_upsample = temperal_upsample

        # dimensions
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        scale = 1.0 / 2**(len(dim_mult) - 2)

        # init block
        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(dims[0], dims[0], dropout), AttentionBlock(dims[0]),
            ResidualBlock(dims[0], dims[0], dropout))

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
                mode = 'upsample3d' if temperal_upsample[i] else 'upsample2d'
                upsamples.append(Resample(out_dim, mode=mode))
                scale *= 2.0
        self.upsamples = nn.Sequential(*upsamples)

        # output blocks
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False), nn.SiLU(),
            CausalConv3d(out_dim, 3, 3, padding=1))

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        # conv1
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([
                    feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                        cache_x.device), cache_x
                ],
                    dim=2)
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
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                            cache_x.device), cache_x
                    ],
                        dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x


def count_conv3d(model):
    count = 0
    for m in model.modules():
        if isinstance(m, CausalConv3d):
            count += 1
    return count


class WanVAE_(nn.Module):

    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_downsample=[True, True, False],
                 dropout=0.0):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.temperal_upsample = temperal_downsample[::-1]

        # modules
        self.encoder = Encoder3d(dim, z_dim * 2, dim_mult, num_res_blocks,
                                 attn_scales, self.temperal_downsample, dropout)
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d(dim, z_dim, dim_mult, num_res_blocks,
                                 attn_scales, self.temperal_upsample, dropout)
        self.clear_cache()

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var

    def encode(self, x, scale):
        self.clear_cache()
        # cache
        t = x.shape[2]
        iter_ = 1 + (t - 1) // 4
        # 对encode输入的x，按时间拆分为1、4、4、4....
        for i in range(iter_):
            self._enc_conv_idx = [0]
            if i == 0:
                out = self.encoder(
                    x[:, :, :1, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx)
            else:
                out_ = self.encoder(
                    x[:, :, 1 + 4 * (i - 1):1 + 4 * i, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx)
                out = torch.cat([out, out_], 2)
        mu, log_var = self.conv1(out).chunk(2, dim=1)
        if isinstance(scale[0], torch.Tensor):
            mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(
                1, self.z_dim, 1, 1, 1)
        else:
            mu = (mu - scale[0]) * scale[1]
        self.clear_cache()
        return mu

    def decode(self, z, scale):
        self.clear_cache()
        # z: [b,c,t,h,w]
        if isinstance(scale[0], torch.Tensor):
            z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(
                1, self.z_dim, 1, 1, 1)
        else:
            z = z / scale[1] + scale[0]
        iter_ = z.shape[2]
        x = self.conv2(z)
        for i in range(iter_):
            self._conv_idx = [0]
            if i == 0:
                out = self.decoder(
                    x[:, :, i:i + 1, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx)
            else:
                out_ = self.decoder(
                    x[:, :, i:i + 1, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx)
                out = torch.cat([out, out_], 2)
        self.clear_cache()
        return out

    def cached_decode(self, z, scale):
        # z: [b,c,t,h,w]
        if isinstance(scale[0], torch.Tensor):
            z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(
                1, self.z_dim, 1, 1, 1)
        else:
            z = z / scale[1] + scale[0]
        iter_ = z.shape[2]
        x = self.conv2(z)
        for i in range(iter_):
            self._conv_idx = [0]
            if i == 0:
                out = self.decoder(
                    x[:, :, i:i + 1, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx)
            else:
                out_ = self.decoder(
                    x[:, :, i:i + 1, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx)
                out = torch.cat([out, out_], 2)
        return out

    def sample(self, imgs, deterministic=False):
        mu, log_var = self.encode(imgs)
        if deterministic:
            return mu
        std = torch.exp(0.5 * log_var.clamp(-30.0, 20.0))
        return mu + std * torch.randn_like(std)

    def clear_cache(self):
        self._conv_num = count_conv3d(self.decoder)
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num
        # cache encode
        self._enc_conv_num = count_conv3d(self.encoder)
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num





class WanVAE:

    def __init__(self,
                 z_dim=16,
                 vae_pth='cache/vae_step_411000.pth',
                 dtype=torch.float,
                 device="cuda"):
        self.dtype = dtype
        self.device = device

        mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]
        self.mean = torch.tensor(mean, dtype=dtype, device=device)
        self.std = torch.tensor(std, dtype=dtype, device=device)
        self.scale = [self.mean, 1.0 / self.std]

        # init model
        self.model = _video_vae(
            pretrained_path=vae_pth,
            z_dim=z_dim,
        ).eval().requires_grad_(False).to(device)

    def encode(self, videos):
        """
        videos: A list of videos each with shape [C, T, H, W].
        """
        with torch.amp.autocast(dtype=self.dtype):
            return [
                self.model.encode(u.unsqueeze(0), self.scale).float().squeeze(0)
                for u in videos
            ]

    def decode(self, zs):
        with torch.amp.autocast(dtype=self.dtype):
            return [
                self.model.decode(u.unsqueeze(0),
                                  self.scale).float().clamp_(-1, 1).squeeze(0)
                for u in zs
            ]


class FeatureCache:
    """Manage cached activations and indices for streaming 3D convolutions."""

    def __init__(self, module: nn.Module):
        self.module = module
        self.num_conv = count_conv3d(module)
        self.clear()

    def clear(self) -> None:
        """Reset cached tensors so subsequent passes start fresh."""
        self._feature_map: List[Optional[torch.Tensor]] = [None] * self.num_conv
        self._indices: List[int] = [0]

    @property
    def feature_map(self) -> List[Optional[torch.Tensor]]:
        return self._feature_map

    def reset_indices(self) -> List[int]:
        """Provide a fresh index list that downstream modules mutate in-place."""
        self._indices = [0]
        return self._indices


class LatentAffineTransform(nn.Module):
    """Apply per-channel affine normalization/denormalization to latents."""

    def __init__(self, z_dim: int):
        super().__init__()
        self.z_dim = z_dim

    def normalize(self, mu: torch.Tensor, scale: List[torch.Tensor]) -> torch.Tensor:
        mean, inv_std = scale
        if isinstance(mean, torch.Tensor):
            mean = mean.view(1, self.z_dim, 1, 1, 1)
            inv_std = inv_std.view(1, self.z_dim, 1, 1, 1)
        return (mu - mean) * inv_std

    def denormalize(self, z: torch.Tensor, scale: List[torch.Tensor]) -> torch.Tensor:
        mean, inv_std = scale
        if isinstance(mean, torch.Tensor):
            mean = mean.view(1, self.z_dim, 1, 1, 1)
            inv_std = inv_std.view(1, self.z_dim, 1, 1, 1)
        return z / inv_std + mean


class LatentStatsProjector(nn.Module):
    """Thin wrapper that maps encoder features to mean/log-variance channels."""

    def __init__(self, z_dim: int):
        super().__init__()
        self.proj = CausalConv3d(z_dim * 2, z_dim * 2, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, log_var = self.proj(x).chunk(2, dim=1)
        return mu, log_var


class LatentInputProjector(nn.Module):
    """Prepare latent inputs for the decoder with a causal 1x1x1 projection."""

    def __init__(self, z_dim: int):
        super().__init__()
        self.proj = CausalConv3d(z_dim, z_dim, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.proj(z)


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
                    latents[:, :, frame_idx:frame_idx + 1, :, :],
                    feat_cache=self.cache.feature_map,
                    feat_idx=self.cache.reset_indices(),
                )
            )
        output = torch.cat(decoded_frames, dim=2)
        if not use_cache:
            self.cache.clear()  # Release cached activations after standalone decode
        return output


class WanVAE_(nn.Module):

    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[True, True, False],
        dropout=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.temperal_upsample = temperal_downsample[::-1]  # Mirror downsample pattern for decoder symmetry

        # Core convolutional backbones
        self.encoder_core = Encoder3d(
            dim, z_dim * 2, dim_mult, num_res_blocks, attn_scales, self.temperal_downsample, dropout
        )
        self.decoder_core = Decoder3d(
            dim, z_dim, dim_mult, num_res_blocks, attn_scales, self.temperal_upsample, dropout
        )

        # Profiling-friendly modular components
        self.latent_affine = LatentAffineTransform(self.z_dim)
        self.latent_stats_projector = LatentStatsProjector(self.z_dim)
        self.latent_input_projector = LatentInputProjector(self.z_dim)
        self.encoder_cache = FeatureCache(self.encoder_core)
        self.decoder_cache = FeatureCache(self.decoder_core)
        self.temporal_encoder = TemporalEncoder(
            self.encoder_core,
            self.latent_stats_projector,
            self.encoder_cache,
            chunk_schedule=(1, 4),
        )
        self.temporal_decoder = TemporalDecoder(
            self.decoder_core,
            self.latent_input_projector,
            self.decoder_cache,
        )
        self.clear_cache()

    def forward(self, x: torch.Tensor, scale: Optional[List[torch.Tensor]] = None):
        mu, log_var = self.encode(x, scale)
        z = self.reparameterize(mu, log_var)  # Draw latent sample via reparameterization trick
        x_recon = self.decode(z, scale)
        return x_recon, mu, log_var

    def encode(self, x: torch.Tensor, scale: Optional[List[torch.Tensor]] = None):
        mu, log_var = self.temporal_encoder(x)
        if scale is not None:
            mu = self.latent_affine.normalize(mu, scale)  # Apply per-channel normalization for downstream diffusion
        return mu, log_var

    def decode(self, z: torch.Tensor, scale: Optional[List[torch.Tensor]] = None):
        if scale is not None:
            z = self.latent_affine.denormalize(z, scale)  # Restore latent statistics to decoder-native range
        out = self.temporal_decoder(z, use_cache=False)
        return out

    def cached_decode(self, z: torch.Tensor, scale: Optional[List[torch.Tensor]] = None):
        if scale is not None:
            z = self.latent_affine.denormalize(z, scale)  # Restore latent statistics before streaming decode
        return self.temporal_decoder(z, use_cache=True)

    def sample(self, imgs: torch.Tensor, deterministic: bool = False, scale: Optional[List[torch.Tensor]] = None):
        mu, log_var = self.encode(imgs, scale)
        if deterministic:
            return mu
        return self.reparameterize(mu, log_var)  # Sample using current mean and log-variance estimates

    def clear_cache(self):
        self.decoder_cache.clear()
        self.encoder_cache.clear()

    @staticmethod
    def reparameterize(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var.clamp(-30.0, 20.0))  # Clamp variance to prevent numerical instability
        return mu + std * torch.randn_like(std)  # Draw sample with reparameterization trick

        
class WanVAEWrapper(torch.nn.Module): # todo

    def __init__(self, device=None):
        super().__init__()
        mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

        # init model
        self.model = self._load_video_vae(
            pretrained_path="skyreels_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
            z_dim=16,
        )
        self.model.eval()
        self.requires_grad_(False) # TODO: is this required here ?
        
    def _load_video_vae(self, pretrained_path=None, z_dim=None, device='cpu', **kwargs):
        """
        Autoencoder3d adapted from Stable Diffusion 1.x, 2.x and XL.
        """
        # params
        cfg = dict(
            dim=96,
            z_dim=z_dim,
            dim_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            attn_scales=[],
            temperal_downsample=[False, True, True],
            dropout=0.0)
        cfg.update(**kwargs)

        # init model
        with torch.device('meta'):
            model = WanVAE_(**cfg)

        model.load_state_dict(
            torch.load(pretrained_path, map_location=device), assign=True)

        return model

    def encode_to_latent(self, pixel: torch.Tensor) -> torch.Tensor:
        # pixel: [batch_size, num_channels, num_frames, height, width]
        device, dtype = pixel.device, pixel.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        output = []
        for frame_seq in pixel:
            latent_stats = self.model.encode(frame_seq.unsqueeze(0), scale)
            if isinstance(latent_stats, tuple):
                latent_stats = latent_stats[0]  # Only retain mean latents for downstream diffusion
            output.append(latent_stats.float().squeeze(0))
        output = torch.stack(output, dim=0)
        return output

    def decode_to_pixel(self, latent: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        if use_cache:
            assert latent.shape[0] == 1, "Batch size must be 1 when using cache"

        device, dtype = latent.device, latent.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        if use_cache:
            decode_function = self.model.cached_decode
        else:
            decode_function = self.model.decode

        output = []
        for latent_seq in latent:
            decoded = decode_function(latent_seq.unsqueeze(0), scale).float().clamp_(-1, 1).squeeze(0)
            output.append(decoded)
        output = torch.stack(output, dim=0)
        return output


class WanDiffusionWrapper(torch.nn.Module): 
    def __init__(
            self,
            model_config="",
            timestep_shift=5.0,
            is_causal=True,
    ):
        super().__init__()
        print(model_config)
        self.model = CausalWanModel.from_config(model_config)
        self.model.eval()

        # For non-causal diffusion, all frames share the same timestep
        self.uniform_timestep = not is_causal

        self.scheduler = FlowMatchScheduler(
            shift=timestep_shift, sigma_min=0.0, extra_one_step=True
        )
        self.scheduler.set_timesteps(1000, training=True)

        self.seq_len = 15 * 880 # 32760  # [1, 15, 16, 60, 104]
        self.post_init()

    def enable_gradient_checkpointing(self) -> None:
        self.model.enable_gradient_checkpointing()

    def _convert_flow_pred_to_x0(self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert flow matching's prediction to x0 prediction.
        flow_pred: the prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = noise - x0
        x_t = (1-sigma_t) * x0 + sigma_t * noise
        we have x0 = x_t - sigma_t * pred
        see derivations https://chatgpt.com/share/67bf8589-3d04-8008-bc6e-4cf1a24e2d0e
        """
        # use higher precision for calculations
        
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device), [flow_pred, xt,
                                                        self.scheduler.sigmas,
                                                        self.scheduler.timesteps]
        )

        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred
        return x0_pred.to(original_dtype)

    @staticmethod
    def _convert_x0_to_flow_pred(scheduler, x0_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert x0 prediction to flow matching's prediction.
        x0_pred: the x0 prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = (x_t - x_0) / sigma_t
        """
        # use higher precision for calculations
        original_dtype = x0_pred.dtype
        x0_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(x0_pred.device), [x0_pred, xt,
                                                      scheduler.sigmas,
                                                      scheduler.timesteps]
        )
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        flow_pred = (xt - x0_pred) / sigma_t
        return flow_pred.to(original_dtype)

    def forward(
        self,
        noisy_image_or_video: torch.Tensor, conditional_dict: dict,
        timestep: torch.Tensor, kv_cache: Optional[List[dict]] = None, kv_cache_mouse: Optional[List[dict]] = None, kv_cache_keyboard: Optional[List[dict]] = None,
        crossattn_cache: Optional[List[dict]] = None,
        current_start: Optional[int] = None,
        cache_start: Optional[int] = None
    ) -> torch.Tensor:
    
        assert noisy_image_or_video.shape[1] == 16
        # [B, F] -> [B]
        if self.uniform_timestep:
            input_timestep = timestep[:, 0]
        else:
            input_timestep = timestep
        logits = None

        if kv_cache is not None:
            flow_pred = self.model(
                noisy_image_or_video.to(self.model.dtype),#.permute(0, 2, 1, 3, 4),
                t=input_timestep, **conditional_dict,
                # seq_len=self.seq_len,
                kv_cache=kv_cache,
                kv_cache_mouse=kv_cache_mouse, kv_cache_keyboard=kv_cache_keyboard,
                crossattn_cache=crossattn_cache,
                current_start=current_start,
                cache_start=cache_start
            )#.permute(0, 2, 1, 3, 4)
            
        else:
            flow_pred = self.model(
                noisy_image_or_video.to(self.model.dtype),#.permute(0, 2, 1, 3, 4),
                t=input_timestep, **conditional_dict)
            #.permute(0, 2, 1, 3, 4)
        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=rearrange(flow_pred, 'b c f h w -> (b f) c h w'),#.flatten(0, 1),
            xt=rearrange(noisy_image_or_video, 'b c f h w -> (b f) c h w'),#.flatten(0, 1),
            timestep=timestep.flatten(0, 1)
        )# .unflatten(0, flow_pred.shape[:2])
        pred_x0 = rearrange(pred_x0, '(b f) c h w -> b c f h w', b=flow_pred.shape[0])
        if logits is not None:
            return flow_pred, pred_x0, logits

        return flow_pred, pred_x0

    def get_scheduler(self) -> SchedulerInterface:
        """
        Update the current scheduler with the interface's static method
        """
        scheduler = self.scheduler
        scheduler.convert_x0_to_noise = types.MethodType(
            SchedulerInterface.convert_x0_to_noise, scheduler)
        scheduler.convert_noise_to_x0 = types.MethodType(
            SchedulerInterface.convert_noise_to_x0, scheduler)
        scheduler.convert_velocity_to_x0 = types.MethodType(
            SchedulerInterface.convert_velocity_to_x0, scheduler)
        self.scheduler = scheduler
        return scheduler

    def post_init(self):
        """
        A few custom initialization steps that should be called after the object is created.
        Currently, the only one we have is to bind a few methods to scheduler.
        We can gradually add more methods here if needed.
        """
        self.get_scheduler()
