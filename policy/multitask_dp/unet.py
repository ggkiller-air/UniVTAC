"""DP ConditionalUnet1D, retaining the reference topology and state-dict layout.

Adapted from real-stanford/diffusion_policy, revision
5ba07ac6661db573af695b419a7947ecb704690f, model/diffusion/{conditional_unet1d,
conv1d_components,positional_embedding}.py. Only unused local conditioning is
omitted. The accompanying DIFFUSION_POLICY_LICENSE covers this implementation.
"""

from __future__ import annotations

import math

import torch
from einops.layers.torch import Rearrange
from torch import Tensor, nn


class SinusoidalPosEmb(nn.Module):
    """Encode scalar diffusion timesteps using the reference frequency spacing."""

    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.dimension = dimension

    def forward(self, timesteps: Tensor) -> Tensor:
        half_dimension = self.dimension // 2
        frequencies = torch.exp(torch.arange(half_dimension, device=timesteps.device)
                                * -(math.log(10000) / (half_dimension - 1)))
        phases = timesteps[:, None] * frequencies[None, :]
        return torch.cat((phases.sin(), phases.cos()), dim=-1)


class Conv1dBlock(nn.Module):
    """Reference convolution, GroupNorm and Mish block."""

    def __init__(self, input_dim: int, output_dim: int, kernel_size: int, n_groups: int = 8) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(input_dim, output_dim, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, output_dim), nn.Mish(),
        )

    def forward(self, sample: Tensor) -> Tensor:
        return self.block(sample)


class Downsample1d(nn.Module):
    """Reference stride-two convolution."""

    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(dimension, dimension, 3, 2, 1)

    def forward(self, sample: Tensor) -> Tensor:
        return self.conv(sample)


class Upsample1d(nn.Module):
    """Reference stride-two transposed convolution."""

    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose1d(dimension, dimension, 4, 2, 1)

    def forward(self, sample: Tensor) -> Tensor:
        return self.conv(sample)


class ConditionalResidualBlock1D(nn.Module):
    """Two convolution blocks with the reference scale/bias FiLM modulation."""

    def __init__(self, input_dim: int, output_dim: int, condition_dim: int, kernel_size: int = 5) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            Conv1dBlock(input_dim, output_dim, kernel_size),
            Conv1dBlock(output_dim, output_dim, kernel_size),
        ])
        self.cond_encoder = nn.Sequential(
            nn.Mish(), nn.Linear(condition_dim, 2 * output_dim), Rearrange("batch features -> batch features 1"),
        )
        self.residual_conv = nn.Conv1d(input_dim, output_dim, 1) if input_dim != output_dim else nn.Identity()
        self.output_dim = output_dim

    def forward(self, sample: Tensor, conditioning: Tensor) -> Tensor:
        features = self.blocks[0](sample)
        modulation = self.cond_encoder(conditioning).reshape(sample.shape[0], 2, self.output_dim, 1)
        features = modulation[:, 0] * features + modulation[:, 1]
        return self.blocks[1](features) + self.residual_conv(sample)


class ConditionalUnet1D(nn.Module):
    """The same global-conditioned U-Net as AdaViP Stage 0; action width is configurable."""

    def __init__(self, input_dim: int = 8, global_cond_dim: int = 256,
                 diffusion_step_embed_dim: int = 128, down_dims: tuple[int, ...] = (512, 1024, 2048)) -> None:
        super().__init__()
        embedding_dim = diffusion_step_embed_dim
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(embedding_dim), nn.Linear(embedding_dim, embedding_dim * 4),
            nn.Mish(), nn.Linear(embedding_dim * 4, embedding_dim),
        )
        condition_dim = embedding_dim + global_cond_dim
        dimension_pairs = list(zip([input_dim, *down_dims[:-1]], down_dims))
        self.down_modules = nn.ModuleList([
            nn.ModuleList([
                ConditionalResidualBlock1D(input_width, output_width, condition_dim),
                ConditionalResidualBlock1D(output_width, output_width, condition_dim),
                Downsample1d(output_width) if index < len(dimension_pairs) - 1 else nn.Identity(),
            ]) for index, (input_width, output_width) in enumerate(dimension_pairs)
        ])
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(down_dims[-1], down_dims[-1], condition_dim),
            ConditionalResidualBlock1D(down_dims[-1], down_dims[-1], condition_dim),
        ])
        self.up_modules = nn.ModuleList([
            nn.ModuleList([
                ConditionalResidualBlock1D(output_width * 2, input_width, condition_dim),
                ConditionalResidualBlock1D(input_width, input_width, condition_dim), Upsample1d(input_width),
            ]) for input_width, output_width in reversed(dimension_pairs[1:])
        ])
        self.final_conv = nn.Sequential(Conv1dBlock(down_dims[0], down_dims[0], 5), nn.Conv1d(down_dims[0], input_dim, 1))

    def forward(self, sample: Tensor, timestep: Tensor | int, global_cond: Tensor) -> Tensor:
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], device=sample.device, dtype=torch.long)
        timesteps = timestep.reshape(-1).expand(sample.shape[0])
        conditioning = torch.cat((self.diffusion_step_encoder(timesteps), global_cond), dim=-1)
        features = sample.transpose(1, 2)
        skips = []
        for first_block, second_block, downsample in self.down_modules:
            features = second_block(first_block(features, conditioning), conditioning)
            skips.append(features)
            features = downsample(features)
        for block in self.mid_modules:
            features = block(features, conditioning)
        for first_block, second_block, upsample in self.up_modules:
            features = torch.cat((features, skips.pop()), dim=1)
            features = upsample(second_block(first_block(features, conditioning), conditioning))
        return self.final_conv(features).transpose(1, 2)
