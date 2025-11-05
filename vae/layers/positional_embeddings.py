"""Positional embedding helpers mirrored from the Wan diffusion stack.

These utilities provide N-D rotary embeddings compatible with the VAE's
spatio-temporal latent grid. They are copied locally so the VAE can operate
without importing the full diffusion model package.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple, Union

import torch


IntOrTuple = Union[int, Iterable[int]]


def _to_tuple(x: IntOrTuple, dim: int = 2) -> Tuple[int, ...]:
    """Convert an integer or iterable into a tuple of length `dim`.

    Args:
        x: Either an integer or an iterable representing sizes.
        dim: Number of dimensions the tuple should cover.

    Returns:
        Tuple[int, ...]: Tuple with `dim` copies of `x` or the original values.

    Raises:
        ValueError: If `x` is an iterable with a length different from `dim`.
    """
    if isinstance(x, int):
        return (x,) * dim
    x = tuple(x)
    if len(x) == dim:
        return x
    raise ValueError(f"Expected length {dim} or int, but got {x}")


def get_meshgrid_nd(start: IntOrTuple, *args: IntOrTuple, dim: int = 2) -> torch.Tensor:
    """Create an N-D meshgrid using semantics similar to `numpy.linspace`.

    Depending on the number of arguments provided, this helper supports
    specifying either the total number of bins, the `[start, stop)` interval,
    or the interval plus an explicit `num` of bins.

    Args:
        start: When used alone, the number of bins per axis. Otherwise the start
            coordinate of the interval.
        *args: Optional values interpreted as `(stop,)` or `(stop, num)`.
        dim: Dimensionality of the resulting meshgrid.

    Returns:
        torch.Tensor: Tensor of shape `[dim, ...]` containing the grid values.

    Raises:
        ValueError: If the number of variadic arguments is not 0, 1, or 2.
    """
    if len(args) == 0:
        # start is grid_size
        num = _to_tuple(start, dim=dim)
        start = (0,) * dim
        stop = num
    elif len(args) == 1:
        # start is start, args[0] is stop, step is 1
        start = _to_tuple(start, dim=dim)
        stop = _to_tuple(args[0], dim=dim)
        num = [stop[i] - start[i] for i in range(dim)]
    elif len(args) == 2:
        # start is start, args[0] is stop, args[1] is num
        start = _to_tuple(start, dim=dim)  # Left-Top       eg: 12,0
        stop = _to_tuple(args[0], dim=dim)  # Right-Bottom   eg: 20,32
        num = _to_tuple(args[1], dim=dim)  # Target Size    eg: 32,124
    else:
        raise ValueError(f"len(args) should be 0, 1 or 2, but got {len(args)}")

    # PyTorch implement of np.linspace(start[i], stop[i], num[i], endpoint=False)
    axis_grid = []
    for i in range(dim):
        a, b, n = start[i], stop[i], num[i]
        g = torch.linspace(
            a, b, n + 1, dtype=torch.float32, device=torch.cuda.current_device()
        )[:n]
        axis_grid.append(g)
    grid = torch.meshgrid(*axis_grid, indexing="ij")  # dim x [W, H, D]
    grid = torch.stack(grid, dim=0)  # [dim, W, H, D]

    return grid


def reshape_for_broadcast(
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    x: torch.Tensor,
    head_first: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Reshape rotary frequencies so they broadcast against `x`.

    Wan uses two attention kernels. With `head_first=True` the tensor layout is
    `[B, S, H, D]`, and with `head_first=False` the head dimension trails the
    spatial dimension (FlashAttention convention).

    Args:
        freqs_cis: Either cosine/sine tensors or a complex rotary embedding.
        x: Target tensor defining the broadcast shape.
        head_first: Whether heads appear before the feature dimension.

    Returns:
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: Rotary embedding
        reshaped to match `x`'s layout.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim

    if isinstance(freqs_cis, tuple):
        # freqs_cis: (cos, sin) in real space
        if head_first:
            assert freqs_cis[0].shape == (
                x.shape[-2],
                x.shape[-1],
            ), f"freqs_cis shape {freqs_cis[0].shape} does not match x shape {x.shape}"
            shape = [
                d if i == ndim - 2 or i == ndim - 1 else 1
                for i, d in enumerate(x.shape)
            ]
        else:
            # assert freqs_cis[0].shape == (
            #     x.shape[1],
            #     x.shape[-1],
            # ), f"freqs_cis shape {freqs_cis[0].shape} does not match x shape {x.shape}"
            # shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
            shape = [1, freqs_cis[0].shape[0], 1, freqs_cis[0].shape[1]]
        return freqs_cis[0].view(*shape), freqs_cis[1].view(*shape)
    else:
        # freqs_cis: values in complex space
        if head_first:
            assert freqs_cis.shape == (
                x.shape[-2],
                x.shape[-1],
            ), f"freqs_cis shape {freqs_cis.shape} does not match x shape {x.shape}"
            shape = [
                d if i == ndim - 2 or i == ndim - 1 else 1
                for i, d in enumerate(x.shape)
            ]
        else:
            assert freqs_cis.shape == (
                x.shape[1],
                x.shape[-1],
            ), f"freqs_cis shape {freqs_cis.shape} does not match x shape {x.shape}"
            shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.view(*shape)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swap even/odd features to simulate multiplication by `i`."""
    x_real, x_imag = (
        x.float().reshape(*x.shape[:-1], -1, 2).unbind(-1)
    )  # [B, S, H, D//2]
    return torch.stack([-x_imag, x_real], dim=-1).flatten(3)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    head_first: bool = False,
    start_offset: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to the provided query/key projections.

    Args:
        xq: Query tensor shaped `[B, S, H, D]` or `[B, H, S, D]`.
        xk: Key tensor with the same layout as `xq`.
        freqs_cis: Rotary embedding represented as complex numbers or `(cos, sin)`.
        head_first: Whether heads precede the feature dimension.
        start_offset: Temporal offset applied when slicing the frequency cache.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Rotary-encoded queries and keys.
    """
    xk_out = None
    assert isinstance(freqs_cis, tuple)
    if isinstance(freqs_cis, tuple):
        cos, sin = reshape_for_broadcast(freqs_cis, xq, head_first)  # [S, D]
        cos, sin = cos.to(xq.device), sin.to(xq.device)
        # real * cos - imag * sin
        # imag * cos + real * sin
        xq_out = (
            xq.float() * cos[:, start_offset : start_offset + xq.shape[1], :, :]
            + rotate_half(xq.float())
            * sin[:, start_offset : start_offset + xq.shape[1], :, :]
        ).type_as(xq)
        xk_out = (
            xk.float() * cos[:, start_offset : start_offset + xk.shape[1], :, :]
            + rotate_half(xk.float())
            * sin[:, start_offset : start_offset + xk.shape[1], :, :]
        ).type_as(xk)
    else:
        # view_as_complex will pack [..., D/2, 2](real) to [..., D/2](complex)
        xq_ = torch.view_as_complex(
            xq.float().reshape(*xq.shape[:-1], -1, 2)
        )  # [B, S, H, D//2]
        freqs_cis = reshape_for_broadcast(freqs_cis, xq_, head_first).to(
            xq.device
        )  # [S, D//2] --> [1, S, 1, D//2]
        # (real, imag) * (cos, sin) = (real * cos - imag * sin, imag * cos + real * sin)
        # view_as_real will expand [..., D/2](complex) to [..., D/2, 2](real)
        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3).type_as(xq)
        xk_ = torch.view_as_complex(
            xk.float().reshape(*xk.shape[:-1], -1, 2)
        )  # [B, S, H, D//2]
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3).type_as(xk)

    return xq_out, xk_out


def get_nd_rotary_pos_embed(
    rope_dim_list: List[int],
    start: IntOrTuple,
    *args: IntOrTuple,
    theta: float = 10000.0,
    use_real: bool = False,
    theta_rescale_factor: Union[float, List[float]] = 1.0,
    interpolation_factor: Union[float, List[float]] = 1.0,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """N-D extension of RoPE tailored to video latents.

    Args:
        rope_dim_list: Per-axis rotary embedding widths that sum to the head dim.
        start: Starting coordinates or number of positions per axis.
        *args: Optional `(stop,)` or `(stop, num)` interval specification.
        theta: Base angular frequency used to build the embedding bank.
        use_real: If True, return `(cos, sin)` instead of complex numbers.
        theta_rescale_factor: Scalar or per-axis rescale factor for `theta`.
        interpolation_factor: Per-axis scaling for the coordinate grid.

    Returns:
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: Rotary embeddings
        flattened over the spatial grid.
    """

    grid = get_meshgrid_nd(
        start, *args, dim=len(rope_dim_list)
    )  # [3, W, H, D] / [2, W, H]

    if isinstance(theta_rescale_factor, int) or isinstance(theta_rescale_factor, float):
        theta_rescale_factor = [theta_rescale_factor] * len(rope_dim_list)
    elif isinstance(theta_rescale_factor, list) and len(theta_rescale_factor) == 1:
        theta_rescale_factor = [theta_rescale_factor[0]] * len(rope_dim_list)
    assert len(theta_rescale_factor) == len(rope_dim_list), (
        "len(theta_rescale_factor) should equal to len(rope_dim_list)"
    )

    if isinstance(interpolation_factor, int) or isinstance(interpolation_factor, float):
        interpolation_factor = [interpolation_factor] * len(rope_dim_list)
    elif isinstance(interpolation_factor, list) and len(interpolation_factor) == 1:
        interpolation_factor = [interpolation_factor[0]] * len(rope_dim_list)
    assert len(interpolation_factor) == len(rope_dim_list), (
        "len(interpolation_factor) should equal to len(rope_dim_list)"
    )

    # use 1/ndim of dimensions to encode grid_axis
    embs = []
    for i in range(len(rope_dim_list)):
        emb = get_1d_rotary_pos_embed(
            rope_dim_list[i],
            grid[i].reshape(-1),
            theta,
            use_real=use_real,
            theta_rescale_factor=theta_rescale_factor[i],
            interpolation_factor=interpolation_factor[i],
        )  # 2 x [WHD, rope_dim_list[i]]
        embs.append(emb)

    if use_real:
        cos = torch.cat([emb[0] for emb in embs], dim=1)  # (WHD, D/2)
        sin = torch.cat([emb[1] for emb in embs], dim=1)  # (WHD, D/2)
        return cos, sin
    else:
        emb = torch.cat(embs, dim=1)  # (WHD, D/2)
        return emb


def get_1d_rotary_pos_embed(
    dim: int,
    pos: Union[torch.FloatTensor, int],
    theta: float = 10000.0,
    use_real: bool = False,
    theta_rescale_factor: float = 1.0,
    interpolation_factor: float = 1.0,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Return 1-D rotary embeddings in complex space or split `(cos, sin)` form.

    Args:
        dim: Feature dimension of the attention head.
        pos: Positions represented as an integer count or explicit tensor.
        theta: Base angular frequency controlling rotation speed.
        use_real: When True, return real cosine/sine tensors.
        theta_rescale_factor: Rescale factor for extending context length.
        interpolation_factor: Multiplicative factor applied to positions.

    Returns:
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: Either complex
        embeddings shaped `[S, D/2]` or real-valued cosine/sine pairs shaped
        `[S, D]`.
    """
    if isinstance(pos, int):
        pos = torch.arange(pos, device=torch.cuda.current_device()).float()

    # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
    # has some connection to NTK literature
    if theta_rescale_factor != 1.0:
        theta *= theta_rescale_factor ** (dim / (dim - 2))

    freqs = 1.0 / (
        theta
        ** (
            torch.arange(0, dim, 2, device=torch.cuda.current_device())[
                : (dim // 2)
            ].float()
            / dim
        )
    )  # [D/2]
    # assert interpolation_factor == 1.0, f"interpolation_factor: {interpolation_factor}"
    freqs = torch.outer(pos * interpolation_factor, freqs)  # [S, D/2]
    if use_real:
        freqs_cos = freqs.cos().repeat_interleave(2, dim=1)  # [S, D]
        freqs_sin = freqs.sin().repeat_interleave(2, dim=1)  # [S, D]
        return freqs_cos, freqs_sin
    else:
        freqs_cis = torch.polar(
            torch.ones_like(freqs), freqs
        )  # complex64     # [S, D/2]
        return freqs_cis


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
        x_i = torch.view_as_complex(
            x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2)
        )
        freqs_i = torch.cat(
            [
                freqs[0][start_frame : start_frame + f]
                .view(f, 1, 1, -1)
                .expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).type_as(x)


def rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    # print(grid_sizes.shape, len(grid_sizes.tolist()), grid_sizes.tolist()[0])
    f, h, w = grid_sizes.tolist()
    for i in range(len(x)):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(
            x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2)
        )
        freqs_i = torch.cat(
            [
                freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).type_as(x)


def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim)),
    )
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half))
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x
