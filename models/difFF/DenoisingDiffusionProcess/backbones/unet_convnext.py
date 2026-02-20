# Code from https://github.com/arpitbansal297/Cold-Diffusion-Models/blob/main/snowification/diffusion/model/unet_convnext.py

from __future__ import annotations

import math
from inspect import isfunction
from typing import Any, Callable, TypeVar

import torch
import torch.nn as nn
from einops import rearrange

T = TypeVar("T")


def exists(x: object) -> bool:
    """Return whether the provided value is not None.

    Args:
        x (object): Input value.

    Returns:
        bool: Computed scalar output.
    """
    return x is not None


def default(val: T | None, d: T | Callable[[], T]) -> T:
    """Return the input value or a fallback default.

    Args:
        val (T | None): Input value.
        d (T | Callable[[], T]): Input value.

    Returns:
        T: Computed output value.
    """
    if exists(val):
        return val
    return d() if isfunction(d) else d


class Residual(nn.Module):
    """Wrapper module that adds a residual skip connection."""
    def __init__(self, fn: nn.Module) -> None:
        """Initialize Residual with configured parameters.

        Args:
            fn (nn.Module): Input value.

        Returns:
            None: No value is returned.
        """
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Run the module forward computation.

        Args:
            x (torch.Tensor): Tensor input for the computation.
            *args (Any): Additional positional arguments forwarded to the underlying call.
            **kwargs (Any): Additional keyword arguments forwarded to the underlying call.

        Returns:
            torch.Tensor: Tensor output produced by this call.
        """
        return self.fn(x, *args, **kwargs) + x


class SinusoidalPosEmb(nn.Module):
    """Module that generates sinusoidal timestep embeddings."""

    def __init__(self, dim: int) -> None:
        """Initialize SinusoidalPosEmb with configured parameters.

        Args:
            dim (int): Input value.

        Returns:
            None: No value is returned.
        """
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the module forward computation.

        Args:
            x (torch.Tensor): Tensor input for the computation.

        Returns:
            torch.Tensor: Tensor output produced by this call.
        """
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def Upsample(dim: int) -> nn.ConvTranspose2d:
    """Create a transpose-convolution upsampling layer.

    Args:
        dim (int): Input value.

    Returns:
        nn.ConvTranspose2d: Computed output value.
    """
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)


def Downsample(dim: int) -> nn.Conv2d:
    """Create a strided-convolution downsampling layer.

    Args:
        dim (int): Input value.

    Returns:
        nn.Conv2d: Computed output value.
    """
    return nn.Conv2d(dim, dim, 4, 2, 1)


class LayerNorm(nn.Module):
    """Channel-wise layer normalization for 2D feature maps."""
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        """Initialize LayerNorm with configured parameters.

        Args:
            dim (int): Input value.
            eps (float): Input value.

        Returns:
            None: No value is returned.
        """
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the module forward computation.

        Args:
            x (torch.Tensor): Tensor input for the computation.

        Returns:
            torch.Tensor: Tensor output produced by this call.
        """
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class PreNorm(nn.Module):
    """Module that normalizes inputs before applying a submodule."""
    def __init__(self, dim: int, fn: nn.Module) -> None:
        """Initialize PreNorm with configured parameters.

        Args:
            dim (int): Input value.
            fn (nn.Module): Input value.

        Returns:
            None: No value is returned.
        """
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the module forward computation.

        Args:
            x (torch.Tensor): Tensor input for the computation.

        Returns:
            torch.Tensor: Tensor output produced by this call.
        """
        x = self.norm(x)
        return self.fn(x)


# building block modules
class ConvNextBlock(nn.Module):
    """ConvNeXt residual block used within the U-Net backbone."""

    def __init__(
        self,
        dim: int,
        dim_out: int,
        *,
        time_emb_dim: int | None = None,
        coord_emb_dim: int | None = None,
        mult: int = 2,
        norm: bool = True,
    ) -> None:
        """Initialize ConvNextBlock with configured parameters.

        Args:
            dim (int): Input value.
            dim_out (int): Input value.
            time_emb_dim (int | None): Input value.
            coord_emb_dim (int | None): Input value.
            mult (int): Input value.
            norm (bool): Boolean flag controlling behavior.

        Returns:
            None: No value is returned.
        """
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.GELU(), nn.Linear(time_emb_dim, dim))
            if exists(time_emb_dim)
            else None
        )
        self.coord_mlp = (
            nn.Sequential(nn.GELU(), nn.Linear(coord_emb_dim, dim * 2))
            if exists(coord_emb_dim)
            else None
        )

        self.ds_conv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)

        self.net = nn.Sequential(
            LayerNorm(dim) if norm else nn.Identity(),
            nn.Conv2d(dim, dim_out * mult, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim_out * mult, dim_out, 3, padding=1),
        )

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor | None = None,
        coord_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the module forward computation.

        Args:
            x (torch.Tensor): Tensor input for the computation.
            time_emb (torch.Tensor | None): Tensor input for the computation.
            coord_emb (torch.Tensor | None): Tensor input for the computation.

        Returns:
            torch.Tensor: Tensor output produced by this call.
        """
        h = self.ds_conv(x)

        if exists(self.mlp):
            assert exists(time_emb), "time emb must be passed in"
            condition = self.mlp(time_emb)
            h = h + rearrange(condition, "b c -> b c 1 1")

        if exists(self.coord_mlp):
            assert exists(coord_emb), "coord emb must be passed in"
            scale_shift = self.coord_mlp(coord_emb)
            scale, shift = scale_shift.chunk(2, dim=1)
            h = h * (1 + rearrange(scale, "b c -> b c 1 1")) + rearrange(
                shift, "b c -> b c 1 1"
            )

        h = self.net(h)
        return h + self.res_conv(x)


class LinearAttention(nn.Module):
    """Linear attention block for efficient spatial mixing."""
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32) -> None:
        """Initialize LinearAttention with configured parameters.

        Args:
            dim (int): Input value.
            heads (int): Input value.
            dim_head (int): Input value.

        Returns:
            None: No value is returned.
        """
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the module forward computation.

        Args:
            x (torch.Tensor): Tensor input for the computation.

        Returns:
            torch.Tensor: Tensor output produced by this call.
        """
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale

        k = k.softmax(dim=-1)
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


# Main Model


class UnetConvNextBlock(nn.Module):
    """U-Net/ConvNeXt backbone used by the diffusion model."""
    def __init__(
        self,
        dim: int,
        out_dim: int | None = None,
        dim_mults: tuple[int, ...] = (1, 2, 4, 8),
        channels: int = 3,
        with_time_emb: bool = True,
        coord_emb_dim: int | None = None,
        output_mean_scale: bool = False,
        residual: bool = False,
    ) -> None:
        """Initialize UnetConvNextBlock with configured parameters.

        Args:
            dim (int): Input value.
            out_dim (int | None): Input value.
            dim_mults (tuple[int, ...]): Input value.
            channels (int): Input value.
            with_time_emb (bool): Boolean flag controlling behavior.
            coord_emb_dim (int | None): Input value.
            output_mean_scale (bool): Boolean flag controlling behavior.
            residual (bool): Boolean flag controlling behavior.

        Returns:
            None: No value is returned.
        """
        super().__init__()
        self.channels = channels
        self.residual = residual
        self.output_mean_scale = output_mean_scale

        dims = [channels, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        if with_time_emb:
            time_dim = dim
            self.time_mlp = nn.Sequential(
                SinusoidalPosEmb(dim),
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim),
            )
        else:
            time_dim = None
            self.time_mlp = None

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        ConvNextBlock(
                            dim_in,
                            dim_out,
                            time_emb_dim=time_dim,
                            coord_emb_dim=coord_emb_dim,
                            norm=ind != 0,
                        ),
                        ConvNextBlock(
                            dim_out, dim_out, time_emb_dim=time_dim, coord_emb_dim=coord_emb_dim
                        ),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = ConvNextBlock(
            mid_dim, mid_dim, time_emb_dim=time_dim, coord_emb_dim=coord_emb_dim
        )
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = ConvNextBlock(
            mid_dim, mid_dim, time_emb_dim=time_dim, coord_emb_dim=coord_emb_dim
        )

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        ConvNextBlock(
                            dim_out * 2,
                            dim_in,
                            time_emb_dim=time_dim,
                            coord_emb_dim=coord_emb_dim,
                        ),
                        ConvNextBlock(
                            dim_in, dim_in, time_emb_dim=time_dim, coord_emb_dim=coord_emb_dim
                        ),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Upsample(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        out_dim = default(out_dim, channels)
        self.final_block = ConvNextBlock(dim, dim, coord_emb_dim=coord_emb_dim)
        self.final_conv = nn.Conv2d(dim, out_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        time: torch.Tensor | None = None,
        coord_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the module forward computation.

        Args:
            x (torch.Tensor): Tensor input for the computation.
            time (torch.Tensor | None): Tensor input for the computation.
            coord_emb (torch.Tensor | None): Tensor input for the computation.

        Returns:
            torch.Tensor: Tensor output produced by this call.
        """
        orig_x = x
        t = None
        if time is not None and exists(self.time_mlp):
            t = self.time_mlp(time)

        original_mean = torch.mean(x, [1, 2, 3], keepdim=True)
        h = []

        for convnext, convnext2, attn, downsample in self.downs:
            x = convnext(x, t, coord_emb)
            x = convnext2(x, t, coord_emb)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t, coord_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t, coord_emb)

        for convnext, convnext2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = convnext(x, t, coord_emb)
            x = convnext2(x, t, coord_emb)
            x = attn(x)
            x = upsample(x)
        if self.residual:
            return self.final_conv(self.final_block(x, coord_emb=coord_emb)) + orig_x

        out = self.final_conv(self.final_block(x, coord_emb=coord_emb))
        if self.output_mean_scale:
            out_mean = torch.mean(out, [1, 2, 3], keepdim=True)
            out = out - original_mean + out_mean

        return out
