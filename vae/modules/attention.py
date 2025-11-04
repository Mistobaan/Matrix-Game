"""Causal attention block used inside the Matrix-Game Wan VAE decoder.

The block mirrors the Wan diffusion backbone: a causal self-attention pass,
optional action conditioning, and a gated MLP. This light-weight wrapper lives
under `vae/` so the VAE stack can reuse Wan attention primitives without
depending on the full diffusion model graph.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
from torch import Tensor, nn

from wan.modules.action_module import ActionModule
from wan.modules.causal_model import CausalWanSelfAttention
from wan.modules.model import WAN_CROSSATTENTION_CLASSES, WanLayerNorm


class CausalWanAttentionBlock(nn.Module):
    """Single Wan attention block with causal masking and action conditioning.

    The block processes latent tokens produced by the Wan VAE. It applies:
      1. LayerNorm + causal self-attention with rotary embeddings.
      2. Cross-attention against encoder / condition context.
      3. Optional mouse / keyboard action injection.
      4. A gated feed-forward network.

    Each sub-layer is modulated by learned scale/shift parameters derived from
    the diffusion timestep embedding (`e` argument in :meth:`forward`).
    """

    def __init__(
        self,
        cross_attn_type: str,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        local_attn_size: int = -1,
        sink_size: int = 0,
        qk_norm: bool = True,
        cross_attn_norm: bool = False,
        action_config: Dict = {},
        block_idx: int = 0,
        eps: float = 1e-6,
    ) -> None:
        """Configure projection sizes, attention flavour, and action hooks.

        Args:
            cross_attn_type: Key inside :data:`WAN_CROSSATTENTION_CLASSES`.
            dim: Channel width of the latent tokens.
            ffn_dim: Hidden size of the MLP.
            num_heads: Attention head count for both self and cross attention.
            local_attn_size: Window size for local causal attention (-1 = global).
            sink_size: Number of persistent memory tokens kept in the cache.
            qk_norm: Whether to apply RMSNorm to Q/K projections.
            cross_attn_norm: Applies LayerNorm before cross-attention if True.
            action_config: Optional dictionary configuring :class:`ActionModule`.
            block_idx: Index of this block, used to enable action modules.
            eps: Numerical stability constant for LayerNorm variants.
        """
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        if len(action_config) != 0 and block_idx in action_config["blocks"]:
            self.action_model = ActionModule(
                **action_config, local_attn_size=self.local_attn_size
            )
        else:
            self.action_model = None

        # Self-attention sub-layer.
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(
            dim, num_heads, local_attn_size, sink_size, qk_norm, eps
        )

        # Cross-attention sub-layer.
        self.norm3 = (
            WanLayerNorm(dim, eps, elementwise_affine=True)
            if cross_attn_norm
            else nn.Identity()
        )
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](
            dim, num_heads, (-1, -1), qk_norm, eps
        )

        # Feed-forward sub-layer.
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )

        # Learned FiLM-style modulation: shape [1, 6, dim] controlling each branch.
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x: Tensor,
        e: Tensor,
        seq_lens: Tensor,
        grid_sizes: Tensor,
        freqs: Tensor,
        context: Tensor,
        block_mask,
        block_mask_mouse,
        block_mask_keyboard,
        num_frame_per_block: int = 3,
        use_rope_keyboard: bool = False,
        mouse_cond: Optional[Tensor] = None,
        keyboard_cond: Optional[Tensor] = None,
        kv_cache: Optional[Dict[str, Tensor]] = None,
        kv_cache_mouse: Optional[Dict[str, Tensor]] = None,
        kv_cache_keyboard: Optional[Dict[str, Tensor]] = None,
        crossattn_cache: Optional[Dict[str, Tensor]] = None,
        current_start: int = 0,
        cache_start: Optional[int] = None,
        context_lens: Optional[Tensor] = None,
    ) -> Tensor:
        """Run the block on a batch of latent tokens.

        Args:
            x: Input tokens `[B, L, C]` where `L = num_frames * frame_seq_len`.
            e: Per-frame modulation tensor `[B, F, 6, C]`.
            seq_lens: Valid token counts for each sample `[B]` (used by flex-attn).
            grid_sizes: Packed `[F, H, W]` describing the 3D latent grid.
            freqs: Rotary frequency cache with shape `[*, C // num_heads // 2]`.
            context: Cross-attention source (e.g. conditioning latents).
            block_mask: FlexAttention `BlockMask` for causal self-attention.
            block_mask_mouse / keyboard: Masks for action conditioning attention.
            num_frame_per_block: Frames covered by a single KV cache segment.
            use_rope_keyboard: Enables rotary embeddings for keyboard conds.
            mouse_cond / keyboard_cond: Optional conditioning streams.
            kv_cache: Self-attention cache for incremental decoding.
            kv_cache_mouse / kv_cache_keyboard: Action-module caches.
            crossattn_cache: Cache for the cross-attention module.
            current_start: Absolute token index processed so far (for caches).
            cache_start: Optional override for cache base index.
            context_lens: Optional valid lengths for the context tensor.

        Returns:
            Tensor: Updated latent tokens `[B, L, C]`.
        """
        assert e.ndim == 4
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]

        # Split modulation tensor into six FiLM gates (shift & scale per sub-layer).
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)

        # 1) Causal self-attention over the latent sequence.
        y = self.self_attn(
            (
                self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen))
                * (1 + e[1])
                + e[0]
            ).flatten(1, 2),
            seq_lens,
            grid_sizes,
            freqs,
            block_mask,
            kv_cache,
            current_start,
            cache_start,
        )

        x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[2]).flatten(
            1, 2
        )

        # 2) Cross-attention + optional action conditioning + gated FFN.
        def cross_attn_ffn(
            x: Tensor,
            context: Tensor,
            e,
            mouse_cond: Optional[Tensor],
            keyboard_cond: Optional[Tensor],
            block_mask_mouse,
            block_mask_keyboard,
            kv_cache_mouse: Optional[Dict[str, Tensor]] = None,
            kv_cache_keyboard: Optional[Dict[str, Tensor]] = None,
            crossattn_cache: Optional[Dict[str, Tensor]] = None,
            start_frame: int = 0,
            use_rope_keyboard: bool = False,
            num_frame_per_block: int = 3,
        ) -> Tensor:
            """Inner helper that applies conditioning modules sequentially."""
            # Cross-attend against the conditioning context (e.g. encodings or prompts).
            x = x + self.cross_attn(
                self.norm3(x.to(context.dtype)),
                context,
                crossattn_cache=crossattn_cache,
            )
            if self.action_model is not None:
                assert mouse_cond is not None or keyboard_cond is not None
                # Inject keyboard/mouse embeddings with their own caches & masks.
                x = self.action_model(
                    x.to(context.dtype),
                    grid_sizes[0],
                    grid_sizes[1],
                    grid_sizes[2],
                    mouse_cond,
                    keyboard_cond,
                    block_mask_mouse,
                    block_mask_keyboard,
                    is_causal=True,
                    kv_cache_mouse=kv_cache_mouse,
                    kv_cache_keyboard=kv_cache_keyboard,
                    start_frame=start_frame,
                    use_rope_keyboard=use_rope_keyboard,
                    num_frame_per_block=num_frame_per_block,
                )

            # Position-wise feed-forward with FiLM gating.
            y = self.ffn(
                (
                    self.norm2(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen))
                    * (1 + e[4])
                    + e[3]
                ).flatten(1, 2)
            )

            x = x + (
                y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[5]
            ).flatten(1, 2)
            return x

        # grid_sizes is stored as a 1-D tensor `[F, H, W]` during cached decoding.
        assert grid_sizes.ndim == 1
        x = cross_attn_ffn(
            x,
            context,
            e,
            mouse_cond,
            keyboard_cond,
            block_mask_mouse,
            block_mask_keyboard,
            kv_cache_mouse,
            kv_cache_keyboard,
            crossattn_cache,
            start_frame=current_start // math.prod(grid_sizes[1:]).item(),
            use_rope_keyboard=use_rope_keyboard,
            num_frame_per_block=num_frame_per_block,
        )
        return x
