import math

import torch
from torch import nn
from torch.nn import functional as functional


class JointConditionHyperNet(nn.Module):
    def __init__(self, feature_dim, state_dim, num_visual_streams, num_tactile_streams,
                 hidden_dim=256, alpha_init=0.01, alpha_max=0.1, use_fusion=True,
                 num_heads=4):
        super().__init__()
        if min(feature_dim, state_dim, num_visual_streams, num_tactile_streams, hidden_dim, num_heads) <= 0:
            raise ValueError("HyperNet dimensions and modality counts must be positive")
        if not 0 < alpha_init < alpha_max:
            raise ValueError("HyperNet requires 0 < alpha_init < alpha_max")
        if use_fusion and feature_dim % num_heads:
            raise ValueError("Feature dimension must be divisible by fusion attention heads")
        self.feature_dim = feature_dim
        self.state_dim = state_dim
        self.num_visual_streams = num_visual_streams
        self.num_streams = num_visual_streams + num_tactile_streams
        self.use_fusion = use_fusion
        self.alpha_max = alpha_max
        self.hypernet = nn.Sequential(
            nn.Linear(self.num_streams * feature_dim + state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * self.num_streams * feature_dim),
        )
        nn.init.zeros_(self.hypernet[-1].weight)
        nn.init.zeros_(self.hypernet[-1].bias)
        alpha_fraction = alpha_init / alpha_max
        self.residual_alpha = nn.Parameter(torch.tensor(math.log(alpha_fraction / (1 - alpha_fraction))))
        if use_fusion:
            self.cross_attention = nn.MultiheadAttention(feature_dim, num_heads, dropout=0.0, batch_first=True)
            self.fusion_gate = nn.Parameter(torch.tensor(0.0))

    @property
    def alpha(self):
        return self.alpha_max * torch.sigmoid(self.residual_alpha)

    def forward(self, features, state):
        if len(features) != self.num_streams:
            raise ValueError(f"Expected {self.num_streams} modality streams, got {len(features)}")
        if state.ndim != 2 or state.shape[-1] != self.state_dim:
            raise ValueError(f"Expected state shape [batch, {self.state_dim}]")
        if any(feature.ndim != 3 or feature.shape[:2] != (state.shape[0], self.feature_dim)
               or feature.shape[-1] == 0 for feature in features):
            raise ValueError("Modality features must have shape [batch, feature_dim, tokens]")
        summaries = [feature.mean(dim=-1) for feature in features]
        context = torch.cat(
            [functional.layer_norm(summary, (self.feature_dim,)) for summary in summaries] + [state],
            dim=-1,
        )
        modulation = torch.tanh(self.hypernet(context)).reshape(
            state.shape[0], self.num_streams, 2, self.feature_dim,
        )
        modulated = [
            feature + self.alpha * (
                modulation[:, stream_index, 0].unsqueeze(-1) * feature
                + modulation[:, stream_index, 1].unsqueeze(-1)
            )
            for stream_index, feature in enumerate(features)
        ]
        if self.use_fusion:
            tokens = torch.stack([feature.mean(dim=-1) for feature in modulated], dim=1)
            attended, _ = self.cross_attention(
                tokens[:, :self.num_visual_streams], tokens, tokens, need_weights=False,
            )
            modulated = [
                feature + torch.tanh(self.fusion_gate) * attended[:, stream_index].unsqueeze(-1)
                if stream_index < self.num_visual_streams else feature
                for stream_index, feature in enumerate(modulated)
            ]
        return modulated
