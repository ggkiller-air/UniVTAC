"""Six-task late-language, full-input DP Teacher for the final UVIP protocol."""

from __future__ import annotations

import copy
import hashlib

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset
from torchvision.models import resnet18

from .data import FullSensorDataset
from .model import FullSensorPolicy, Normalizer, replace_batch_norm
from .schema import IMAGE_KEYS, SENSOR_KEYS, fingerprint


FINAL_TASKS = ("insert_hole", "insert_tube", "lift_bottle", "lift_can",
               "pull_out_key", "put_bottle_in_shelf")


class SixTaskDataset(Dataset):
    """Select complete episodes and fit statistics using only the final six tasks."""

    def __init__(self, source: FullSensorDataset) -> None:
        self.source = source
        task_ids = [source.manifest["tasks"].index(task) for task in FINAL_TASKS]
        episodes = [e for e in source.manifest["episodes"] if e["task_index"] in task_ids]
        if {e["task"] for e in episodes} != set(FINAL_TASKS):
            raise ValueError("The six selected tasks must all have episodes")
        self.indices = np.concatenate([np.arange(e["start"], e["end"], dtype=np.int64) for e in episodes])
        self.task_indices = source.task_indices[self.indices]
        counts = np.bincount(self.task_indices)
        self.sample_weights = torch.from_numpy(1.0 / counts[self.task_indices])
        self.manifest = {"source_cache_id": source.manifest["cache_id"], "tasks": list(FINAL_TASKS),
                         "episodes": episodes, "total_steps": len(self.indices),
                         "source_indices_sha256": hashlib.sha256(self.indices.tobytes()).hexdigest()}
        self.manifest["cache_id"] = fingerprint(self.manifest)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict:
        return self.source[int(self.indices[index])]

    def epoch_indices(self, seed: int, epoch: int) -> list[int]:
        """Keep replacement sampling balanced and reproducible across resume."""
        generator = torch.Generator().manual_seed(seed + epoch)
        return torch.multinomial(self.sample_weights, len(self), True, generator=generator).tolist()

    def normalization_statistics(self) -> dict:
        """Exclude both omitted tasks from state/action normalization."""
        result = {}
        for name in ("state", "action"):
            values = self.source.arrays[name][self.indices]
            result[name] = {"min": torch.from_numpy(values.min(0)), "max": torch.from_numpy(values.max(0))}
        return result


class L1SensorEncoder(nn.Module):
    """Five non-language tokens -> 256D state, then [state,CLIP512] -> condition."""

    def __init__(self, history: int = 2, token_dim: int = 256,
                 backbone: nn.Module | None = None, image_feature_dim: int = 512) -> None:
        super().__init__()
        self.history, self.token_dim = history, token_dim
        if backbone is None:
            backbone = resnet18(weights=None)
            backbone.fc = nn.Identity()
            replace_batch_norm(backbone)
        self.image_encoders = nn.ModuleDict({key: copy.deepcopy(backbone) for key in IMAGE_KEYS})
        self.temporal_projections = nn.ModuleDict({key: nn.Sequential(
            nn.Linear(history * image_feature_dim, token_dim), nn.LayerNorm(token_dim), nn.SiLU(),
        ) for key in IMAGE_KEYS})
        self.state_projection = nn.Sequential(nn.Linear(history * 8, token_dim), nn.LayerNorm(token_dim), nn.SiLU())
        self.sensor_identity = nn.Parameter(torch.empty(5, token_dim))
        nn.init.normal_(self.sensor_identity, std=.02)
        self.fusion = nn.Sequential(nn.Linear(5 * token_dim + 5, token_dim), nn.SiLU(), nn.Linear(token_dim, token_dim))
        self.language_head = nn.Sequential(nn.Linear(token_dim + 512, token_dim), nn.SiLU(), nn.Linear(token_dim, token_dim))

    def encode_tokens(self, observations: dict[str, Tensor], normalizer: Normalizer) -> Tensor:
        """Normalize each uint8 image once and keep language out of sensor tokens."""
        if set(observations) != set(SENSOR_KEYS):
            raise ValueError("Expected four images, state and language")
        tokens = []
        for key in IMAGE_KEYS:
            images = observations[key]
            if images.dtype != torch.uint8 or images.shape[1:3] != (self.history, 3):
                raise ValueError(f"Invalid uint8 image history: {key}")
            features = self.image_encoders[key](images.flatten(0, 1).float().mul_(2 / 255).sub_(1))
            tokens.append(self.temporal_projections[key](features.reshape(len(images), -1)))
        tokens.append(self.state_projection(normalizer.normalize("state", observations["state"]).flatten(1)))
        return torch.stack(tokens, 1) + self.sensor_identity[None]

    def condition(self, tokens: Tensor, language: Tensor, presence: Tensor) -> Tensor:
        """Use exact zero projected tokens/identities and zero late language for scoring."""
        if presence.dtype != torch.bool or presence.shape != (len(tokens), 6):
            raise ValueError("Expected six boolean presence flags")
        if tokens.shape[1:] != (5, self.token_dim) or language.shape != (len(tokens), 512):
            raise ValueError("Invalid L1 token or language dimensions")
        masked = torch.where(presence[:, :5, None], tokens, torch.zeros_like(tokens))
        z_t = self.fusion(torch.cat((masked.flatten(1), presence[:, :5].to(tokens.dtype)), -1))
        late_language = torch.where(presence[:, 5:6], language, torch.zeros_like(language))
        return self.language_head(torch.cat((z_t, late_language), -1))


class L1Policy(FullSensorPolicy):
    """Action-only L1 with no training dropout and native explicit eval interventions."""

    def __init__(self, config: dict, encoder: L1SensorEncoder | None = None) -> None:
        if config["full_input_probability"] != 1.0:
            raise ValueError("Final L1 Teacher requires full inputs with no dropout")
        super().__init__(config, encoder or L1SensorEncoder(config["n_obs_steps"], config["token_dim"]))

    def compute_loss_per_sample(self, batch: dict, condition_ids: Tensor | None = None,
                                denoising_sample=None) -> Tensor:
        batch_size = len(batch["action"])
        if condition_ids is None:
            condition_ids = torch.zeros(batch_size, device=batch["action"].device, dtype=torch.long)
        if condition_ids.shape != (batch_size,) or condition_ids.dtype != torch.long:
            raise ValueError("Expected int64 condition ids")
        if torch.any(condition_ids < 0) or torch.any(condition_ids >= len(self.condition_presence)):
            raise ValueError("Unknown L1 intervention")
        if self.training and condition_ids.any():
            raise ValueError("Final L1 training forbids input dropout")
        tokens = self.obs_encoder.encode_tokens(batch["obs"], self.normalizer)
        conditioning = self.obs_encoder.condition(tokens, batch["obs"]["language"][:, 0],
                                                   self.condition_presence[condition_ids])
        sample = self.prepare_denoising_sample(batch) if denoising_sample is None else denoising_sample
        return self.denoising_loss_per_sample(sample, conditioning)

    @torch.no_grad()
    def paired_denoising_losses(self, batch: dict, noise=None, timesteps=None) -> dict[str, Tensor]:
        if self.training:
            raise RuntimeError("Paired scoring requires eval mode")
        tokens = self.obs_encoder.encode_tokens(batch["obs"], self.normalizer)
        sample = self.prepare_denoising_sample(batch, noise, timesteps)
        losses = torch.stack([self.denoising_loss_per_sample(sample, self.obs_encoder.condition(
            tokens, batch["obs"]["language"][:, 0], flags.expand(len(tokens), -1)))
            for flags in self.condition_presence], 1)
        return {"losses": losses, "utility": losses[:, 1:] - losses[:, :1]}

    @torch.no_grad()
    def predict_action(self, obs_dict: dict[str, Tensor], presence=None) -> dict[str, Tensor]:
        tokens = self.obs_encoder.encode_tokens(obs_dict, self.normalizer)
        flags = self.condition_presence[0].expand(len(tokens), -1) if presence is None else presence
        conditioning = self.obs_encoder.condition(tokens, obs_dict["language"][:, 0], flags)
        trajectory = torch.randn(len(tokens), self.horizon, 8, device=tokens.device)
        self.noise_scheduler.set_timesteps(self.config["num_inference_steps"])
        for timestep in self.noise_scheduler.timesteps:
            prediction = self.model(trajectory, timestep, global_cond=conditioning)
            trajectory = self.noise_scheduler.step(prediction, timestep, trajectory).prev_sample
        actions = self.normalizer.unnormalize("action", trajectory)
        start = self.n_obs_steps - 1
        return {"action": actions[:, start:start + self.n_action_steps], "action_pred": actions}
