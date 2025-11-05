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

# from wan.modules.attention import attention
# from wan.modules.model import (
#     MLPProj,
#     WanLayerNorm,
#     rope_apply,
#     rope_params,
#     sinusoidal_embedding_1d,
# )
# from .modules.positional_embeddings import (
#     rope_apply,
#     rope_params,
#     sinuisoidal_embedding_1d,
# )
import math
import types
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.modeling_utils import ModelMixin
from einops import rearrange
from torch import nn
from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask,
)

from utils.scheduler import FlowMatchScheduler, SchedulerInterface

from .blocks.attention import CausalWanAttentionBlock
from .blocks.decoder import TemporalDecoder
from .blocks.encoder import TemporalEncoder
from .blocks.head import CausalHead
from .layers.positional_embeddings import rope_params, sinusoidal_embedding_1d


from typing import List

import tensorrt as trt
import torch
import torch.nn as nn
from einops import rearrange

from demo_utils.constant import ALL_INPUTS_NAMES, ZERO_VAE_CACHE

__all__ = [
    "WanVAE",
    "CausalWanModel",
]

CACHE_T = 2


# class CausalWanModel(ModelMixin, ConfigMixin, FromOriginalModelMixin, PeftAdapterMixin):
#     r"""
#     Wan diffusion backbone supporting both text-to-video and image-to-video.
#     """

#     ignore_for_config = ["patch_size", "cross_attn_norm", "qk_norm", "text_dim"]
#     _no_split_modules = ["WanAttentionBlock"]
#     _supports_gradient_checkpointing = True

#     @register_to_config
#     def __init__(
#         self,
#         model_type="t2v",
#         patch_size=(1, 2, 2),
#         text_len=512,
#         in_dim=36,
#         dim=1536,
#         ffn_dim=8960,
#         freq_dim=256,
#         text_dim=4096,
#         out_dim=16,
#         num_heads=12,
#         num_layers=30,
#         local_attn_size=-1,
#         sink_size=0,
#         qk_norm=True,
#         cross_attn_norm=True,
#         action_config={},
#         eps=1e-6,
#     ):
#         r"""
#         Initialize the diffusion model backbone.

#         Args:
#             model_type (`str`, *optional*, defaults to 't2v'):
#                 Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
#             patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
#                 3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
#             text_len (`int`, *optional*, defaults to 512):
#                 Fixed length for text embeddings
#             in_dim (`int`, *optional*, defaults to 16):
#                 Input video channels (C_in)
#             dim (`int`, *optional*, defaults to 2048):
#                 Hidden dimension of the transformer
#             ffn_dim (`int`, *optional*, defaults to 8192):
#                 Intermediate dimension in feed-forward network
#             freq_dim (`int`, *optional*, defaults to 256):
#                 Dimension for sinusoidal time embeddings
#             text_dim (`int`, *optional*, defaults to 4096):
#                 Input dimension for text embeddings
#             out_dim (`int`, *optional*, defaults to 16):
#                 Output video channels (C_out)
#             num_heads (`int`, *optional*, defaults to 16):
#                 Number of attention heads
#             num_layers (`int`, *optional*, defaults to 32):
#                 Number of transformer blocks
#             local_attn_size (`int`, *optional*, defaults to -1):
#                 Window size for temporal local attention (-1 indicates global attention)
#             sink_size (`int`, *optional*, defaults to 0):
#                 Size of the attention sink, we keep the first `sink_size` frames unchanged when rolling the KV cache
#             qk_norm (`bool`, *optional*, defaults to True):
#                 Enable query/key normalization
#             cross_attn_norm (`bool`, *optional*, defaults to False):
#                 Enable cross-attention normalization
#             eps (`float`, *optional*, defaults to 1e-6):
#                 Epsilon value for normalization layers
#         """

#         super().__init__()

#         assert model_type in ["i2v"]
#         self.model_type = model_type
#         self.use_action_module = len(action_config) > 0
#         self.patch_size = patch_size
#         self.text_len = text_len
#         self.in_dim = in_dim
#         self.dim = dim
#         self.ffn_dim = ffn_dim
#         self.freq_dim = freq_dim
#         self.text_dim = text_dim
#         self.out_dim = out_dim
#         self.num_heads = num_heads
#         self.num_layers = num_layers
#         self.local_attn_size = local_attn_size
#         self.qk_norm = qk_norm
#         self.cross_attn_norm = cross_attn_norm
#         self.eps = eps

#         # embeddings
#         self.patch_embedding = nn.Conv3d(
#             in_channels=in_dim,
#             out_channels=dim,
#             kernel_size=patch_size,
#             stride=patch_size,
#         )

#         self.time_embedding = nn.Sequential(
#             nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
#         )

#         TIME_PROJECTION_CONST = 6
#         self.time_projection = nn.Sequential(
#             nn.SiLU(), nn.Linear(dim, dim * TIME_PROJECTION_CONST)
#         )

#         # blocks
#         # cross_attn_type = "i2v_cross_attn"
#         self.blocks = nn.ModuleList(
#             [
#                 CausalWanAttentionBlock(
#                     dim,
#                     ffn_dim,
#                     num_heads,
#                     local_attn_size,
#                     sink_size,
#                     qk_norm,
#                     cross_attn_norm,
#                     action_config=action_config,
#                     eps=eps,
#                     block_idx=idx,
#                 )
#                 for idx in range(num_layers)
#             ]
#         )

#         # head
#         self.head = CausalHead(dim, out_dim, patch_size, eps)

#         # buffers (don't use register_buffer otherwise dtype will be changed in to())
#         assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
#         d = dim // num_heads

#         self.freqs = torch.cat(
#             [
#                 rope_params(1024, d - 4 * (d // 6)),
#                 rope_params(1024, 2 * (d // 6)),
#                 rope_params(1024, 2 * (d // 6)),
#             ],
#             dim=1,
#         )

#         if model_type == "i2v":
#             self.img_emb = MLPProj(1280, dim)

#         # initialize weights
#         self.init_weights()

#         self.gradient_checkpointing = False

#         self.block_mask = None
#         self.block_mask_keyboard = None
#         self.block_mask_mouse = None
#         self.use_rope_keyboard = True
#         self.num_frame_per_block = 1

#     def _set_gradient_checkpointing(self, module, value=False):
#         self.gradient_checkpointing = value

#     @staticmethod
#     def _prepare_blockwise_causal_attn_mask(
#         device: torch.device | str,
#         num_frames: int = 9,
#         frame_seqlen: int = 880,
#         num_frame_per_block=1,
#         local_attn_size=-1,
#     ) -> BlockMask:
#         """
#         we will divide the token sequence into the following format
#         [1 latent frame] [1 latent frame] ... [1 latent frame]
#         We use flexattention to construct the attention mask
#         """
#         total_length = num_frames * frame_seqlen

#         # we do right padding to get to a multiple of 128
#         padded_length = math.ceil(total_length / 128) * 128 - total_length

#         ends = torch.zeros(
#             total_length + padded_length, device=device, dtype=torch.long
#         )

#         # Block-wise causal mask will attend to all elements that are before the end of the current chunk
#         frame_indices = torch.arange(
#             start=0,
#             end=total_length,
#             step=frame_seqlen * num_frame_per_block,
#             device=device,
#         )

#         for tmp in frame_indices:
#             ends[tmp : tmp + frame_seqlen * num_frame_per_block] = (
#                 tmp + frame_seqlen * num_frame_per_block
#             )

#         def attention_mask(b, h, q_idx, kv_idx):
#             if local_attn_size == -1:
#                 return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
#             else:
#                 return (
#                     (kv_idx < ends[q_idx])
#                     & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))
#                 ) | (q_idx == kv_idx)

#         block_mask = create_block_mask(
#             attention_mask,
#             B=None,
#             H=None,
#             Q_LEN=total_length + padded_length,
#             KV_LEN=total_length + padded_length,
#             _compile=False,
#             device=device,
#         )

#         import torch.distributed as dist

#         if not dist.is_initialized() or dist.get_rank() == 0:
#             print(
#                 f" cache a block wise causal mask with block size of {num_frame_per_block} frames"
#             )

#         return block_mask

#     @staticmethod
#     def _prepare_blockwise_causal_attn_mask_keyboard(
#         device: torch.device | str,
#         num_frames: int = 9,
#         frame_seqlen: int = 880,
#         num_frame_per_block=1,
#         local_attn_size=-1,
#     ) -> BlockMask:
#         """
#         we will divide the token sequence into the following format
#         [1 latent frame] [1 latent frame] ... [1 latent frame]
#         We use flexattention to construct the attention mask
#         """
#         total_length2 = num_frames * frame_seqlen

#         # we do right padding to get to a multiple of 128
#         padded_length2 = math.ceil(total_length2 / 32) * 32 - total_length2
#         padded_length_kv2 = math.ceil(num_frames / 32) * 32 - num_frames
#         ends2 = torch.zeros(
#             total_length2 + padded_length2, device=device, dtype=torch.long
#         )

#         # Block-wise causal mask will attend to all elements that are before the end of the current chunk
#         frame_indices2 = torch.arange(
#             start=0,
#             end=total_length2,
#             step=frame_seqlen * num_frame_per_block,
#             device=device,
#         )
#         cnt = num_frame_per_block
#         for tmp in frame_indices2:
#             ends2[tmp : tmp + frame_seqlen * num_frame_per_block] = cnt
#             cnt += num_frame_per_block

#         def attention_mask2(b, h, q_idx, kv_idx):
#             if local_attn_size == -1:
#                 return (kv_idx < ends2[q_idx]) | (q_idx == kv_idx)
#             else:
#                 return (
#                     (kv_idx < ends2[q_idx])
#                     & (kv_idx >= (ends2[q_idx] - local_attn_size))
#                 ) | (q_idx == kv_idx)

#         block_mask2 = create_block_mask(
#             attention_mask2,
#             B=None,
#             H=None,
#             Q_LEN=total_length2 + padded_length2,
#             KV_LEN=num_frames + padded_length_kv2,
#             _compile=False,
#             device=device,
#         )

#         import torch.distributed as dist

#         if not dist.is_initialized() or dist.get_rank() == 0:
#             print(
#                 f" cache a block wise causal mask with block size of {num_frame_per_block} frames"
#             )

#         return block_mask2

#     @staticmethod
#     def _prepare_blockwise_causal_attn_mask_action(
#         device: torch.device | str,
#         num_frames: int = 9,
#         frame_seqlen: int = 1,
#         num_frame_per_block=1,
#         local_attn_size=-1,
#     ) -> BlockMask:
#         """
#         we will divide the token sequence into the following format
#         [1 latent frame] [1 latent frame] ... [1 latent frame]
#         We use flexattention to construct the attention mask
#         """
#         total_length2 = num_frames * frame_seqlen

#         # we do right padding to get to a multiple of 128
#         padded_length2 = math.ceil(total_length2 / 32) * 32 - total_length2
#         padded_length_kv2 = math.ceil(num_frames / 32) * 32 - num_frames
#         ends2 = torch.zeros(
#             total_length2 + padded_length2, device=device, dtype=torch.long
#         )

#         # Block-wise causal mask will attend to all elements that are before the end of the current chunk
#         frame_indices2 = torch.arange(
#             start=0,
#             end=total_length2,
#             step=frame_seqlen * num_frame_per_block,
#             device=device,
#         )
#         cnt = num_frame_per_block
#         for tmp in frame_indices2:
#             ends2[tmp : tmp + frame_seqlen * num_frame_per_block] = cnt
#             cnt += num_frame_per_block

#         def attention_mask2(b, h, q_idx, kv_idx):
#             if local_attn_size == -1:
#                 return (kv_idx < ends2[q_idx]) | (q_idx == kv_idx)
#             else:
#                 return (
#                     (kv_idx < ends2[q_idx])
#                     & (kv_idx >= (ends2[q_idx] - local_attn_size))
#                 ) | (q_idx == kv_idx)

#         block_mask2 = create_block_mask(
#             attention_mask2,
#             B=None,
#             H=None,
#             Q_LEN=total_length2 + padded_length2,
#             KV_LEN=num_frames + padded_length_kv2,
#             _compile=False,
#             device=device,
#         )

#         import torch.distributed as dist

#         if not dist.is_initialized() or dist.get_rank() == 0:
#             print(
#                 f" cache a block wise causal mask with block size of {num_frame_per_block} frames"
#             )

#         return block_mask2

#     def _forward_inference(
#         self,
#         x,
#         t,
#         visual_context,
#         cond_concat,
#         mouse_cond=None,
#         keyboard_cond=None,
#         kv_cache: dict = None,
#         kv_cache_mouse=None,
#         kv_cache_keyboard=None,
#         crossattn_cache: dict = None,
#         current_start: int = 0,
#         cache_start: int = 0,
#     ):
#         r"""
#         Run the diffusion model with kv caching.
#         See Algorithm 2 of CausVid paper https://arxiv.org/abs/2412.07772 for details.
#         This function will be run for num_frame times.
#         Process the latent frames one by one (1560 tokens each)

#         Args:
#             x (List[Tensor]):
#                 List of input video tensors, each with shape [C_in, F, H, W]
#             t (Tensor):
#                 Diffusion timesteps tensor of shape [B]
#             context (List[Tensor]):
#                 List of text embeddings each with shape [L, C]
#             seq_len (`int`):
#                 Maximum sequence length for positional encoding
#             clip_fea (Tensor, *optional*):
#                 CLIP image features for image-to-video mode
#             y (List[Tensor], *optional*):
#                 Conditional video inputs for image-to-video mode, same shape as x

#         Returns:
#             List[Tensor]:
#                 List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
#         """

#         if mouse_cond is not None or keyboard_cond is not None:
#             assert self.use_action_module is True
#         # params
#         device = self.patch_embedding.weight.device
#         if self.freqs.device != device:
#             self.freqs = self.freqs.to(device)

#         x = torch.cat([x, cond_concat], dim=1)  # B C' F H W

#         # embeddings
#         x = self.patch_embedding(x)
#         grid_sizes = torch.tensor(x.shape[2:], dtype=torch.long)
#         x = x.flatten(2).transpose(1, 2)  # B FHW C'
#         seq_lens = torch.tensor([u.size(0) for u in x], dtype=torch.long)
#         assert seq_lens[0] <= 15 * 1 * 880

#         e = self.time_embedding(
#             sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x)
#         )
#         e0 = (
#             self.time_projection(e)
#             .unflatten(1, (6, self.dim))
#             .unflatten(dim=0, sizes=t.shape)
#         )
#         # context
#         context_lens = None
#         context = self.img_emb(visual_context)
#         # arguments
#         kwargs = dict(
#             e=e0,
#             seq_lens=seq_lens,
#             grid_sizes=grid_sizes,
#             freqs=self.freqs,
#             context=context,
#             mouse_cond=mouse_cond,
#             context_lens=context_lens,
#             keyboard_cond=keyboard_cond,
#             block_mask=self.block_mask,
#             block_mask_mouse=self.block_mask_mouse,
#             block_mask_keyboard=self.block_mask_keyboard,
#             use_rope_keyboard=self.use_rope_keyboard,
#             num_frame_per_block=self.num_frame_per_block,
#         )

#         def create_custom_forward(module):
#             def custom_forward(*inputs, **kwargs):
#                 return module(*inputs, **kwargs)

#             return custom_forward

#         for block_index, block in enumerate(self.blocks):
#             if torch.is_grad_enabled() and self.gradient_checkpointing:
#                 kwargs.update(
#                     {
#                         "kv_cache": kv_cache[block_index],
#                         "kv_cache_mouse": kv_cache_mouse[block_index],
#                         "kv_cache_keyboard": kv_cache_keyboard[block_index],
#                         "current_start": current_start,
#                         "cache_start": cache_start,
#                     }
#                 )
#                 x = torch.utils.checkpoint.checkpoint(
#                     create_custom_forward(block),
#                     x,
#                     **kwargs,
#                     use_reentrant=False,
#                 )
#             else:
#                 kwargs.update(
#                     {
#                         "kv_cache": kv_cache[block_index],
#                         "kv_cache_mouse": kv_cache_mouse[block_index],
#                         "kv_cache_keyboard": kv_cache_keyboard[block_index],
#                         "crossattn_cache": crossattn_cache[block_index],
#                         "current_start": current_start,
#                         "cache_start": cache_start,
#                     }
#                 )
#                 x = block(x, **kwargs)

#         # head
#         x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
#         # unpatchify
#         x = self.unpatchify(x, grid_sizes)
#         return x

#     def _forward_train(
#         self,
#         x,
#         t,
#         visual_context,
#         cond_concat,
#         mouse_cond=None,
#         keyboard_cond=None,
#     ):
#         r"""
#         Forward pass through the diffusion model

#         Args:
#             x (List[Tensor]):
#                 List of input video tensors, each with shape [C_in, F, H, W]
#             t (Tensor):
#                 Diffusion timesteps tensor of shape [B]
#             context (List[Tensor]):
#                 List of text embeddings each with shape [L, C]
#             seq_len (`int`):
#                 Maximum sequence length for positional encoding
#             clip_fea (Tensor, *optional*):
#                 CLIP image features for image-to-video mode
#             y (List[Tensor], *optional*):
#                 Conditional video inputs for image-to-video mode, same shape as x

#         Returns:
#             List[Tensor]:
#                 List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
#         """
#         # params
#         if mouse_cond is not None or keyboard_cond is not None:
#             assert self.use_action_module is True
#         device = self.patch_embedding.weight.device
#         if self.freqs.device != device:
#             self.freqs = self.freqs.to(device)
#         x = torch.cat([x, cond_concat], dim=1)
#         # Construct blockwise causal attn mask
#         if self.block_mask is None:
#             self.block_mask = self._prepare_blockwise_causal_attn_mask(
#                 device,
#                 num_frames=x.shape[2],
#                 frame_seqlen=x.shape[-2]
#                 * x.shape[-1]
#                 // (self.patch_size[1] * self.patch_size[2]),
#                 num_frame_per_block=self.num_frame_per_block,
#                 local_attn_size=self.local_attn_size,
#             )
#         if self.block_mask_keyboard is None:
#             if self.use_rope_keyboard is False:
#                 self.block_mask_keyboard = (
#                     self._prepare_blockwise_causal_attn_mask_keyboard(
#                         device,
#                         num_frames=x.shape[2],
#                         frame_seqlen=x.shape[-2]
#                         * x.shape[-1]
#                         // (self.patch_size[1] * self.patch_size[2]),
#                         num_frame_per_block=self.num_frame_per_block,
#                         local_attn_size=self.local_attn_size,
#                     )
#                 )
#             else:
#                 self.block_mask_keyboard = (
#                     self._prepare_blockwise_causal_attn_mask_action(
#                         device,
#                         num_frames=x.shape[2],
#                         frame_seqlen=1,
#                         num_frame_per_block=self.num_frame_per_block,
#                         local_attn_size=self.local_attn_size,
#                     )
#                 )
#         if self.block_mask_mouse is None:
#             self.block_mask_mouse = self._prepare_blockwise_causal_attn_mask_action(
#                 device,
#                 num_frames=x.shape[2],
#                 frame_seqlen=1,
#                 num_frame_per_block=self.num_frame_per_block,
#                 local_attn_size=self.local_attn_size,
#             )
#         x = self.patch_embedding(x)
#         grid_sizes = torch.tensor(x.shape[2:], dtype=torch.long)
#         x = x.flatten(2).transpose(1, 2)
#         seq_lens = torch.tensor([u.size(0) for u in x], dtype=torch.long)
#         e = self.time_embedding(
#             sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x)
#         )
#         e0 = (
#             self.time_projection(e)
#             .unflatten(1, (6, self.dim))
#             .unflatten(dim=0, sizes=t.shape)
#         )

#         context_lens = None
#         context = self.img_emb(visual_context)

#         # arguments
#         kwargs = dict(
#             e=e0,
#             seq_lens=seq_lens,
#             grid_sizes=grid_sizes,
#             freqs=self.freqs,
#             context=context,
#             mouse_cond=mouse_cond,
#             context_lens=context_lens,
#             keyboard_cond=keyboard_cond,
#             block_mask=self.block_mask,
#             block_mask_mouse=self.block_mask_mouse,
#             block_mask_keyboard=self.block_mask_keyboard,
#             use_rope_keyboard=self.use_rope_keyboard,
#             num_frame_per_block=self.num_frame_per_block,
#         )

#         def create_custom_forward(module):
#             def custom_forward(*inputs, **kwargs):
#                 return module(*inputs, **kwargs)

#             return custom_forward

#         for block in self.blocks:
#             if torch.is_grad_enabled() and self.gradient_checkpointing:
#                 x = torch.utils.checkpoint.checkpoint(
#                     create_custom_forward(block),
#                     x,
#                     **kwargs,
#                     use_reentrant=False,
#                 )
#             else:
#                 x = block(x, **kwargs)

#         # head
#         x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
#         # unpatchify
#         x = self.unpatchify(x, grid_sizes)
#         return x

#     def forward(self, *args, **kwargs):
#         if kwargs.get("kv_cache", None) is not None:
#             return self._forward_inference(*args, **kwargs)
#         else:
#             return self._forward_train(*args, **kwargs)

#     def unpatchify(self, x, grid_sizes):
#         r"""
#         Reconstruct video tensors from patch embeddings.

#         Args:
#             x (List[Tensor]):
#                 List of patchified features, each with shape [L, C_out * prod(patch_size)]
#             grid_sizes (Tensor):
#                 Original spatial-temporal grid dimensions before patching,
#                     shape [3] (3 dimensions correspond to F_patches, H_patches, W_patches)

#         Returns:
#             List[Tensor]:
#                 Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
#         """

#         c = self.out_dim
#         bs = x.shape[0]
#         x = x.view(bs, *grid_sizes, *self.patch_size, c)
#         x = torch.einsum("bfhwpqrc->bcfphqwr", x)
#         x = x.reshape(bs, c, *[i * j for i, j in zip(grid_sizes, self.patch_size)])
#         return x

#     def init_weights(self):
#         r"""
#         Initialize model parameters using Xavier initialization.
#         """

#         # basic init
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)

#         # init embeddings
#         nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))

#         for m in self.time_embedding.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, std=0.02)

#         # init output layer
#         nn.init.zeros_(self.head.head.weight)
#         if self.use_action_module is True:
#             for m in self.blocks:
#                 try:
#                     nn.init.zeros_(m.action_model.proj_mouse.weight)
#                     if m.action_model.proj_mouse.bias is not None:
#                         nn.init.zeros_(m.action_model.proj_mouse.bias)
#                     nn.init.zeros_(m.action_model.proj_keyboard.weight)
#                     if m.action_model.proj_keyboard.bias is not None:
#                         nn.init.zeros_(m.action_model.proj_keyboard.bias)
#                 except Exception:
#                     pass


from .layers.convolution import CausalConv3d
from .blocks.encoder import Encoder3d
from .blocks.decoder import Decoder3d


def count_conv3d(model):
    count = 0
    for m in model.modules():
        if isinstance(m, CausalConv3d):
            count += 1
    return count


class WanVAE_(nn.Module):
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
        self.temperal_upsample = temporal_downsample[::-1]

        # modules
        self.encoder = Encoder3d(
            dim,
            z_dim * 2,
            dim_mult,
            num_res_blocks,
            attn_scales,
            self.temporal_downsample,
            dropout,
        )
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d(
            dim,
            z_dim,
            dim_mult,
            num_res_blocks,
            attn_scales,
            self.temperal_upsample,
            dropout,
        )
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
        # For the encode input x, split it along the time axis into chunks of sizes 1, 4, 4, 4, ...
        for i in range(iter_):
            self._enc_conv_idx = [0]
            if i == 0:
                out = self.encoder(
                    x[:, :, :1, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx,
                )
            else:
                out_ = self.encoder(
                    x[:, :, 1 + 4 * (i - 1) : 1 + 4 * i, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx,
                )
                out = torch.cat([out, out_], 2)
        mu, log_var = self.conv1(out).chunk(2, dim=1)
        if isinstance(scale[0], torch.Tensor):
            mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(
                1, self.z_dim, 1, 1, 1
            )
        else:
            mu = (mu - scale[0]) * scale[1]
        self.clear_cache()
        return mu

    def decode(self, z, scale):
        self.clear_cache()
        # z: [b,c,t,h,w]
        if isinstance(scale[0], torch.Tensor):
            z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(
                1, self.z_dim, 1, 1, 1
            )
        else:
            z = z / scale[1] + scale[0]
        iter_ = z.shape[2]
        x = self.conv2(z)
        for i in range(iter_):
            self._conv_idx = [0]
            if i == 0:
                out = self.decoder(
                    x[:, :, i : i + 1, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx,
                )
            else:
                out_ = self.decoder(
                    x[:, :, i : i + 1, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx,
                )
                out = torch.cat([out, out_], 2)
        self.clear_cache()
        return out

    def cached_decode(self, z, scale):
        # z: [b,c,t,h,w]
        if isinstance(scale[0], torch.Tensor):
            z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(
                1, self.z_dim, 1, 1, 1
            )
        else:
            z = z / scale[1] + scale[0]
        iter_ = z.shape[2]
        x = self.conv2(z)
        for i in range(iter_):
            self._conv_idx = [0]
            if i == 0:
                out = self.decoder(
                    x[:, :, i : i + 1, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx,
                )
            else:
                out_ = self.decoder(
                    x[:, :, i : i + 1, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx,
                )
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


# class WanVAE:
#     def __init__(
#         self,
#         z_dim=16,
#         vae_pth="cache/vae_step_411000.pth",
#         dtype=torch.float,
#         device="cuda",
#     ):
#         self.dtype = dtype
#         self.device = device

#         mean = [
#             -0.7571,
#             -0.7089,
#             -0.9113,
#             0.1075,
#             -0.1745,
#             0.9653,
#             -0.1517,
#             1.5508,
#             0.4134,
#             -0.0715,
#             0.5517,
#             -0.3632,
#             -0.1922,
#             -0.9497,
#             0.2503,
#             -0.2921,
#         ]
#         std = [
#             2.8184,
#             1.4541,
#             2.3275,
#             2.6558,
#             1.2196,
#             1.7708,
#             2.6052,
#             2.0743,
#             3.2687,
#             2.1526,
#             2.8652,
#             1.5579,
#             1.6382,
#             1.1253,
#             2.8251,
#             1.9160,
#         ]
#         self.mean = torch.tensor(mean, dtype=dtype, device=device)
#         self.std = torch.tensor(std, dtype=dtype, device=device)
#         self.scale = [self.mean, 1.0 / self.std]

#         # init model
#         self.model = (
#             _video_vae(
#                 pretrained_path=vae_pth,
#                 z_dim=z_dim,
#             )
#             .eval()
#             .requires_grad_(False)
#             .to(device)
#         )

#     def encode(self, videos):
#         """
#         videos: A list of videos each with shape [C, T, H, W].
#         """
#         with torch.amp.autocast(dtype=self.dtype):
#             return [
#                 self.model.encode(u.unsqueeze(0), self.scale).float().squeeze(0)
#                 for u in videos
#             ]

#     def decode(self, zs):
#         with torch.amp.autocast(dtype=self.dtype):
#             return [
#                 self.model.decode(u.unsqueeze(0), self.scale)
#                 .float()
#                 .clamp_(-1, 1)
#                 .squeeze(0)
#                 for u in zs
#             ]


# class WanVAE_(nn.Module):
#     def __init__(
#         self,
#         dim=128,
#         z_dim=4,
#         dim_mult=[1, 2, 4, 4],
#         num_res_blocks=2,
#         attn_scales=[],
#         temporal_downsample=[True, True, False],
#         dropout=0.0,
#     ):
#         super().__init__()
#         self.dim = dim
#         self.z_dim = z_dim
#         self.dim_mult = dim_mult
#         self.num_res_blocks = num_res_blocks
#         self.attn_scales = attn_scales
#         self.temporal_downsample = temporal_downsample
#         self.temperal_upsample = temporal_downsample[
#             ::-1
#         ]  # Mirror downsample pattern for decoder symmetry

#         # Core convolutional backbones
#         self.encoder_core = Encoder3d(
#             dim,
#             z_dim * 2,
#             dim_mult,
#             num_res_blocks,
#             attn_scales,
#             self.temporal_downsample,
#             dropout,
#         )
#         self.decoder_core = Decoder3d(
#             dim,
#             z_dim,
#             dim_mult,
#             num_res_blocks,
#             attn_scales,
#             self.temperal_upsample,
#             dropout,
#         )

#         # Profiling-friendly modular components
#         self.latent_affine = LatentAffineTransform(self.z_dim)
#         self.latent_stats_projector = LatentStatsProjector(self.z_dim)
#         self.latent_input_projector = LatentInputProjector(self.z_dim)
#         self.encoder_cache = FeatureCache(self.encoder_core)
#         self.decoder_cache = FeatureCache(self.decoder_core)
#         self.temporal_encoder = TemporalEncoder(
#             self.encoder_core,
#             self.latent_stats_projector,
#             self.encoder_cache,
#             chunk_schedule=(1, 4),
#         )
#         self.temporal_decoder = TemporalDecoder(
#             self.decoder_core,
#             self.latent_input_projector,
#             self.decoder_cache,
#         )
#         self.clear_cache()

#     def forward(self, x: torch.Tensor, scale: Optional[List[torch.Tensor]] = None):
#         mu, log_var = self.encode(x, scale)
#         z = self.reparameterize(
#             mu, log_var
#         )  # Draw latent sample via reparameterization trick
#         x_recon = self.decode(z, scale)
#         return x_recon, mu, log_var

#     def encode(self, x: torch.Tensor, scale: Optional[List[torch.Tensor]] = None):
#         mu, log_var = self.temporal_encoder(x)
#         if scale is not None:
#             mu = self.latent_affine.normalize(
#                 mu, scale
#             )  # Apply per-channel normalization for downstream diffusion
#         return mu, log_var

#     def decode(self, z: torch.Tensor, scale: Optional[List[torch.Tensor]] = None):
#         if scale is not None:
#             z = self.latent_affine.denormalize(
#                 z, scale
#             )  # Restore latent statistics to decoder-native range
#         out = self.temporal_decoder(z, use_cache=False)
#         return out

#     def cached_decode(
#         self, z: torch.Tensor, scale: Optional[List[torch.Tensor]] = None
#     ):
#         if scale is not None:
#             z = self.latent_affine.denormalize(
#                 z, scale
#             )  # Restore latent statistics before streaming decode
#         return self.temporal_decoder(z, use_cache=True)

#     def sample(
#         self,
#         imgs: torch.Tensor,
#         deterministic: bool = False,
#         scale: Optional[List[torch.Tensor]] = None,
#     ):
#         mu, log_var = self.encode(imgs, scale)
#         if deterministic:
#             return mu
#         return self.reparameterize(
#             mu, log_var
#         )  # Sample using current mean and log-variance estimates

#     def clear_cache(self):
#         self.decoder_cache.clear()
#         self.encoder_cache.clear()

#     @staticmethod
#     def reparameterize(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
#         std = torch.exp(
#             0.5 * log_var.clamp(-30.0, 20.0)
#         )  # Clamp variance to prevent numerical instability
#         return mu + std * torch.randn_like(
#             std
#         )  # Draw sample with reparameterization trick


class WanVAEWrapper(torch.nn.Module):  # todo
    def __init__(self, pretrained_path=None, z_dim=None, device="cpu"):
        super().__init__()
        mean = [
            -0.7571,
            -0.7089,
            -0.9113,
            0.1075,
            -0.1745,
            0.9653,
            -0.1517,
            1.5508,
            0.4134,
            -0.0715,
            0.5517,
            -0.3632,
            -0.1922,
            -0.9497,
            0.2503,
            -0.2921,
        ]
        std = [
            2.8184,
            1.4541,
            2.3275,
            2.6558,
            1.2196,
            1.7708,
            2.6052,
            2.0743,
            3.2687,
            2.1526,
            2.8652,
            1.5579,
            1.6382,
            1.1253,
            2.8251,
            1.9160,
        ]
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

        # init model
        self.model = self._load_video_vae(
            pretrained_path="skyreels_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
            z_dim=16,
        )
        self.model.eval()
        self.requires_grad_(False)  # TODO: is this required here ?

    def _load_video_vae(self, pretrained_path=None, z_dim=None, device="cpu", **kwargs):
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
            temporal_downsample=[False, True, True],
            dropout=0.0,
        )
        cfg.update(**kwargs)

        # init model
        with torch.device("meta"):
            model = WanVAE_(**cfg)

        model.load_state_dict(
            torch.load(pretrained_path, map_location=device), assign=True
        )

        return model

    def encode_to_latent(self, pixel: torch.Tensor) -> torch.Tensor:
        # pixel: [batch_size, num_channels, num_frames, height, width]
        device, dtype = pixel.device, pixel.dtype
        scale = [
            self.mean.to(device=device, dtype=dtype),
            1.0 / self.std.to(device=device, dtype=dtype),
        ]

        output = []
        for frame_seq in pixel:
            latent_stats = self.model.encode(frame_seq.unsqueeze(0), scale)
            if isinstance(latent_stats, tuple):
                latent_stats = latent_stats[
                    0
                ]  # Only retain mean latents for downstream diffusion
            output.append(latent_stats.float().squeeze(0))
        output = torch.stack(output, dim=0)
        return output

    def decode_to_pixel(
        self, latent: torch.Tensor, use_cache: bool = False
    ) -> torch.Tensor:
        if use_cache:
            assert latent.shape[0] == 1, "Batch size must be 1 when using cache"

        device, dtype = latent.device, latent.dtype
        scale = [
            self.mean.to(device=device, dtype=dtype),
            1.0 / self.std.to(device=device, dtype=dtype),
        ]

        if use_cache:
            decode_function = self.model.cached_decode
        else:
            decode_function = self.model.decode

        output = []
        for latent_seq in latent:
            decoded = (
                decode_function(latent_seq.unsqueeze(0), scale)
                .float()
                .clamp_(-1, 1)
                .squeeze(0)
            )
            output.append(decoded)
        output = torch.stack(output, dim=0)
        return output


# class WanDiffusionWrapper(torch.nn.Module):
#     def __init__(
#         self,
#         model_config="",
#         timestep_shift=5.0,
#         is_causal=True,
#     ):
#         super().__init__()
#         print(model_config)
#         self.model = CausalWanModel.from_config(model_config)
#         self.model.eval()

#         # For non-causal diffusion, all frames share the same timestep
#         self.uniform_timestep = not is_causal

#         self.scheduler = FlowMatchScheduler(
#             shift=timestep_shift, sigma_min=0.0, extra_one_step=True
#         )
#         self.scheduler.set_timesteps(1000, training=True)

#         self.seq_len = 15 * 880  # 32760  # [1, 15, 16, 60, 104]
#         self.post_init()

#     def enable_gradient_checkpointing(self) -> None:
#         self.model.enable_gradient_checkpointing()

#     def _convert_flow_pred_to_x0(
#         self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor
#     ) -> torch.Tensor:
#         """
#         Convert flow matching's prediction to x0 prediction.
#         flow_pred: the prediction with shape [B, C, H, W]
#         xt: the input noisy data with shape [B, C, H, W]
#         timestep: the timestep with shape [B]

#         pred = noise - x0
#         x_t = (1-sigma_t) * x0 + sigma_t * noise
#         we have x0 = x_t - sigma_t * pred
#         see derivations https://chatgpt.com/share/67bf8589-3d04-8008-bc6e-4cf1a24e2d0e
#         """
#         # use higher precision for calculations

#         original_dtype = flow_pred.dtype
#         flow_pred, xt, sigmas, timesteps = map(
#             lambda x: x.double().to(flow_pred.device),
#             [flow_pred, xt, self.scheduler.sigmas, self.scheduler.timesteps],
#         )

#         timestep_id = torch.argmin(
#             (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1
#         )
#         sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
#         x0_pred = xt - sigma_t * flow_pred
#         return x0_pred.to(original_dtype)

#     @staticmethod
#     def _convert_x0_to_flow_pred(
#         scheduler, x0_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor
#     ) -> torch.Tensor:
#         """
#         Convert x0 prediction to flow matching's prediction.
#         x0_pred: the x0 prediction with shape [B, C, H, W]
#         xt: the input noisy data with shape [B, C, H, W]
#         timestep: the timestep with shape [B]

#         pred = (x_t - x_0) / sigma_t
#         """
#         # use higher precision for calculations
#         original_dtype = x0_pred.dtype
#         x0_pred, xt, sigmas, timesteps = map(
#             lambda x: x.double().to(x0_pred.device),
#             [x0_pred, xt, scheduler.sigmas, scheduler.timesteps],
#         )
#         timestep_id = torch.argmin(
#             (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1
#         )
#         sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
#         flow_pred = (xt - x0_pred) / sigma_t
#         return flow_pred.to(original_dtype)

#     def forward(
#         self,
#         noisy_image_or_video: torch.Tensor,
#         conditional_dict: dict,
#         timestep: torch.Tensor,
#         kv_cache: Optional[List[dict]] = None,
#         kv_cache_mouse: Optional[List[dict]] = None,
#         kv_cache_keyboard: Optional[List[dict]] = None,
#         crossattn_cache: Optional[List[dict]] = None,
#         current_start: Optional[int] = None,
#         cache_start: Optional[int] = None,
#     ) -> torch.Tensor:
#         assert noisy_image_or_video.shape[1] == 16
#         # [B, F] -> [B]
#         if self.uniform_timestep:
#             input_timestep = timestep[:, 0]
#         else:
#             input_timestep = timestep
#         logits = None

#         if kv_cache is not None:
#             flow_pred = self.model(
#                 noisy_image_or_video.to(self.model.dtype),  # .permute(0, 2, 1, 3, 4),
#                 t=input_timestep,
#                 **conditional_dict,
#                 # seq_len=self.seq_len,
#                 kv_cache=kv_cache,
#                 kv_cache_mouse=kv_cache_mouse,
#                 kv_cache_keyboard=kv_cache_keyboard,
#                 crossattn_cache=crossattn_cache,
#                 current_start=current_start,
#                 cache_start=cache_start,
#             )  # .permute(0, 2, 1, 3, 4)

#         else:
#             flow_pred = self.model(
#                 noisy_image_or_video.to(self.model.dtype),  # .permute(0, 2, 1, 3, 4),
#                 t=input_timestep,
#                 **conditional_dict,
#             )
#             # .permute(0, 2, 1, 3, 4)
#         pred_x0 = self._convert_flow_pred_to_x0(
#             flow_pred=rearrange(
#                 flow_pred, "b c f h w -> (b f) c h w"
#             ),  # .flatten(0, 1),
#             xt=rearrange(
#                 noisy_image_or_video, "b c f h w -> (b f) c h w"
#             ),  # .flatten(0, 1),
#             timestep=timestep.flatten(0, 1),
#         )  # .unflatten(0, flow_pred.shape[:2])
#         pred_x0 = rearrange(pred_x0, "(b f) c h w -> b c f h w", b=flow_pred.shape[0])
#         if logits is not None:
#             return flow_pred, pred_x0, logits

#         return flow_pred, pred_x0

#     def get_scheduler(self) -> SchedulerInterface:
#         """
#         Update the current scheduler with the interface's static method
#         """
#         scheduler = self.scheduler
#         scheduler.convert_x0_to_noise = types.MethodType(
#             SchedulerInterface.convert_x0_to_noise, scheduler
#         )
#         scheduler.convert_noise_to_x0 = types.MethodType(
#             SchedulerInterface.convert_noise_to_x0, scheduler
#         )
#         scheduler.convert_velocity_to_x0 = types.MethodType(
#             SchedulerInterface.convert_velocity_to_x0, scheduler
#         )
#         self.scheduler = scheduler
#         return scheduler

#     def post_init(self):
#         """
#         A few custom initialization steps that should be called after the object is created.
#         Currently, the only one we have is to bind a few methods to scheduler.
#         We can gradually add more methods here if needed.
#         """
#         self.get_scheduler()


# from wan.modules.vae import CausalConv3d, RMS_norm, Upsample

CACHE_T = 2


# class VAEDecoderWrapperSingle(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.decoder = VAEDecoder3d()
#         mean = [
#             -0.7571,
#             -0.7089,
#             -0.9113,
#             0.1075,
#             -0.1745,
#             0.9653,
#             -0.1517,
#             1.5508,
#             0.4134,
#             -0.0715,
#             0.5517,
#             -0.3632,
#             -0.1922,
#             -0.9497,
#             0.2503,
#             -0.2921,
#         ]
#         std = [
#             2.8184,
#             1.4541,
#             2.3275,
#             2.6558,
#             1.2196,
#             1.7708,
#             2.6052,
#             2.0743,
#             3.2687,
#             2.1526,
#             2.8652,
#             1.5579,
#             1.6382,
#             1.1253,
#             2.8251,
#             1.9160,
#         ]
#         self.mean = torch.tensor(mean, dtype=torch.float32)
#         self.std = torch.tensor(std, dtype=torch.float32)
#         self.z_dim = 16
#         self.conv2 = CausalConv3d(self.z_dim, self.z_dim, 1)

#     def forward(
#         self,
#         z: torch.Tensor,
#         is_first_frame: torch.Tensor,
#         *feat_cache: List[torch.Tensor],
#     ):
#         # from [batch_size, num_frames, num_channels, height, width]
#         # to [batch_size, num_channels, num_frames, height, width]
#         z = z.permute(0, 2, 1, 3, 4)
#         assert z.shape[2] == 1
#         feat_cache = list(feat_cache)
#         is_first_frame = is_first_frame.bool()

#         device, dtype = z.device, z.dtype
#         scale = [
#             self.mean.to(device=device, dtype=dtype),
#             1.0 / self.std.to(device=device, dtype=dtype),
#         ]

#         if isinstance(scale[0], torch.Tensor):
#             z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(
#                 1, self.z_dim, 1, 1, 1
#             )
#         else:
#             z = z / scale[1] + scale[0]
#         x = self.conv2(z)
#         out, feat_cache = self.decoder(x, is_first_frame, feat_cache=feat_cache)
#         out = out.clamp_(-1, 1)
#         # from [batch_size, num_channels, num_frames, height, width]
#         # to [batch_size, num_frames, num_channels, height, width]
#         out = out.permute(0, 2, 1, 3, 4)
#         return out, feat_cache


# class VAEDecoder3d(nn.Module):
#     def __init__(
#         self,
#         dim=96,
#         z_dim=16,
#         dim_mult=[1, 2, 4, 4],
#         num_res_blocks=2,
#         attn_scales=[],
#         temperal_upsample=[True, True, False],
#         dropout=0.0,
#     ):
#         super().__init__()
#         self.dim = dim
#         self.z_dim = z_dim
#         self.dim_mult = dim_mult
#         self.num_res_blocks = num_res_blocks
#         self.attn_scales = attn_scales
#         self.temperal_upsample = temperal_upsample
#         self.cache_t = 2
#         self.decoder_conv_num = 32

#         # dimensions
#         dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
#         scale = 1.0 / 2 ** (len(dim_mult) - 2)

#         # init block
#         self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)

#         # middle blocks
#         self.middle = nn.Sequential(
#             ResidualBlock(dims[0], dims[0], dropout),
#             AttentionBlock(dims[0]),
#             ResidualBlock(dims[0], dims[0], dropout),
#         )

#         # upsample blocks
#         upsamples = []
#         for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
#             # residual (+attention) blocks
#             if i == 1 or i == 2 or i == 3:
#                 in_dim = in_dim // 2
#             for _ in range(num_res_blocks + 1):
#                 upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
#                 if scale in attn_scales:
#                     upsamples.append(AttentionBlock(out_dim))
#                 in_dim = out_dim

#             # upsample block
#             if i != len(dim_mult) - 1:
#                 mode = "upsample3d" if temperal_upsample[i] else "upsample2d"
#                 upsamples.append(Resample(out_dim, mode=mode))
#                 scale *= 2.0
#         self.upsamples = nn.Sequential(*upsamples)

#         # output blocks
#         self.head = nn.Sequential(
#             RMS_norm(out_dim, images=False),
#             nn.SiLU(),
#             CausalConv3d(out_dim, 3, 3, padding=1),
#         )

#     def forward(
#         self,
#         x: torch.Tensor,
#         is_first_frame: torch.Tensor,
#         feat_cache: List[torch.Tensor],
#     ):
#         idx = 0
#         out_feat_cache = []

#         # conv1
#         cache_x = x[:, :, -self.cache_t :, :, :].clone()
#         if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
#             # cache last frame of last two chunk
#             cache_x = torch.cat(
#                 [
#                     feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device),
#                     cache_x,
#                 ],
#                 dim=2,
#             )
#         x = self.conv1(x, feat_cache[idx])
#         out_feat_cache.append(cache_x)
#         idx += 1

#         # middle
#         for layer in self.middle:
#             if isinstance(layer, ResidualBlock) and feat_cache is not None:
#                 x, out_feat_cache_1, out_feat_cache_2 = layer(
#                     x, feat_cache[idx], feat_cache[idx + 1]
#                 )
#                 idx += 2
#                 out_feat_cache.append(out_feat_cache_1)
#                 out_feat_cache.append(out_feat_cache_2)
#             else:
#                 x = layer(x)

#         # upsamples
#         for layer in self.upsamples:
#             if isinstance(layer, Resample):
#                 x, cache_x = layer(x, is_first_frame, feat_cache[idx])
#                 if cache_x is not None:
#                     out_feat_cache.append(cache_x)
#                     idx += 1
#             else:
#                 x, out_feat_cache_1, out_feat_cache_2 = layer(
#                     x, feat_cache[idx], feat_cache[idx + 1]
#                 )
#                 idx += 2
#                 out_feat_cache.append(out_feat_cache_1)
#                 out_feat_cache.append(out_feat_cache_2)

#         # head
#         for layer in self.head:
#             if isinstance(layer, CausalConv3d) and feat_cache is not None:
#                 cache_x = x[:, :, -self.cache_t :, :, :].clone()
#                 if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
#                     # cache last frame of last two chunk
#                     cache_x = torch.cat(
#                         [
#                             feat_cache[idx][:, :, -1, :, :]
#                             .unsqueeze(2)
#                             .to(cache_x.device),
#                             cache_x,
#                         ],
#                         dim=2,
#                     )
#                 x = layer(x, feat_cache[idx])
#                 out_feat_cache.append(cache_x)
#                 idx += 1
#             else:
#                 x = layer(x)
#         return x, out_feat_cache


# class VAETRTWrapper:
#     def __init__(self):
#         TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
#         with (
#             open("checkpoints/vae_decoder_int8.trt", "rb") as f,
#             trt.Runtime(TRT_LOGGER) as rt,
#         ):
#             self.engine: trt.ICudaEngine = rt.deserialize_cuda_engine(f.read())

#         self.context: trt.IExecutionContext = self.engine.create_execution_context()
#         self.stream = torch.cuda.current_stream().cuda_stream

#         # ──────────────────────────────
#         # 2️⃣  Feed the engine with tensors
#         #     (name-based API in TRT ≥10)
#         # ──────────────────────────────
#         self.dtype_map = {
#             trt.float32: torch.float32,
#             trt.float16: torch.float16,
#             trt.int8: torch.int8,
#             trt.int32: torch.int32,
#         }
#         test_input = torch.zeros(1, 16, 1, 60, 104).cuda().half()
#         is_first_frame = torch.tensor(1.0).cuda().half()
#         test_cache_inputs = [c.cuda().half() for c in ZERO_VAE_CACHE]
#         test_inputs = [test_input, is_first_frame] + test_cache_inputs

#         # keep references so buffers stay alive
#         self.device_buffers, self.outputs = {}, []

#         # ---- inputs ----
#         for i, name in enumerate(ALL_INPUTS_NAMES):
#             tensor, scale = test_inputs[i], 1 / 127
#             tensor = self.quantize_if_needed(
#                 tensor, self.engine.get_tensor_dtype(name), scale
#             )

#             # dynamic shapes
#             if -1 in self.engine.get_tensor_shape(name):
#                 # new API :contentReference[oaicite:0]{index=0}
#                 self.context.set_input_shape(name, tuple(tensor.shape))

#             # replaces bindings[] :contentReference[oaicite:1]{index=1}
#             self.context.set_tensor_address(name, int(tensor.data_ptr()))
#             self.device_buffers[name] = tensor  # keep pointer alive

#         # ---- (after all input shapes are known) infer output shapes ----
#         # propagates shapes :contentReference[oaicite:2]{index=2}
#         self.context.infer_shapes()

#         for i in range(self.engine.num_io_tensors):
#             name = self.engine.get_tensor_name(i)
#             # replaces binding_is_input :contentReference[oaicite:3]{index=3}
#             if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
#                 shape = tuple(self.context.get_tensor_shape(name))
#                 dtype = self.dtype_map[self.engine.get_tensor_dtype(name)]
#                 out = torch.empty(shape, dtype=dtype, device="cuda").contiguous()

#                 self.context.set_tensor_address(name, int(out.data_ptr()))
#                 self.outputs.append(out)
#                 self.device_buffers[name] = out

#     # helper to quant-convert on the fly
#     def quantize_if_needed(self, t, expected_dtype, scale):
#         if expected_dtype == trt.int8 and t.dtype != torch.int8:
#             t = torch.clamp((t / scale).round(), -128, 127).to(torch.int8).contiguous()
#         return t  # keep pointer alive

#     def forward(self, *test_inputs):
#         for i, name in enumerate(ALL_INPUTS_NAMES):
#             tensor, scale = test_inputs[i], 1 / 127
#             tensor = self.quantize_if_needed(
#                 tensor, self.engine.get_tensor_dtype(name), scale
#             )
#             self.context.set_tensor_address(name, int(tensor.data_ptr()))
#             self.device_buffers[name] = tensor

#         self.context.execute_async_v3(stream_handle=self.stream)
#         torch.cuda.current_stream().synchronize()
#         return self.outputs
