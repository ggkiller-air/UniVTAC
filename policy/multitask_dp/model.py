"""Independent Stage-0 full-sensor DDPM with paired per-sample denoising losses."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch import Tensor, nn
from torchvision.models import resnet18

from .schema import CONDITION_NAMES, IMAGE_KEYS, SENSOR_KEYS, condition_probabilities, presence_table
from .unet import ConditionalUnet1D


class Normalizer(nn.Module):
    """Checkpointed limits normalization; constant channels have unit scale."""

    def __init__(self) -> None:
        super().__init__()
        for key in ("state", "action"):
            self.register_buffer(f"{key}_scale", torch.ones(8))
            self.register_buffer(f"{key}_offset", torch.zeros(8))
            self.register_buffer(f"{key}_min", torch.zeros(8))
            self.register_buffer(f"{key}_max", torch.zeros(8))

    @torch.no_grad()
    def fit(self, statistics: dict) -> None:
        """Fit once from training rows and preserve the original limits in state_dict."""
        for key in ("state", "action"):
            minimum, maximum = statistics[key]["min"], statistics[key]["max"]
            span = maximum - minimum
            constant = span < 1e-4
            scale = torch.where(constant, torch.ones_like(span), 2 / span.clamp_min(1e-4))
            offset = torch.where(constant, -minimum, -1 - scale * minimum)
            for name, value in (("scale", scale), ("offset", offset), ("min", minimum), ("max", maximum)):
                getattr(self, f"{key}_{name}").copy_(value)

    def normalize(self, key: str, values: Tensor) -> Tensor:
        return values.float() * getattr(self, f"{key}_scale") + getattr(self, f"{key}_offset")

    def unnormalize(self, key: str, values: Tensor) -> Tensor:
        return (values - getattr(self, f"{key}_offset")) / getattr(self, f"{key}_scale")


def replace_batch_norm(module: nn.Module) -> None:
    """Match the AdaViP ResNet18 BatchNorm-to-GroupNorm substitution."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, nn.GroupNorm(math.gcd(32, child.num_features), child.num_features))
        else:
            replace_batch_norm(child)


class SensorEncoder(nn.Module):
    """One independent ResNet18 per real image sensor; temporal compression precedes fusion."""

    def __init__(self, history: int = 2, token_dim: int = 256, backbone: nn.Module | None = None,
                 image_feature_dim: int = 512) -> None:
        super().__init__()
        if backbone is None:
            backbone = resnet18(weights=None)
            backbone.fc = nn.Identity()
            replace_batch_norm(backbone)
        self.history, self.token_dim = history, token_dim
        self.image_encoders = nn.ModuleDict({key: copy.deepcopy(backbone) for key in IMAGE_KEYS})
        self.temporal_projections = nn.ModuleDict({key: nn.Sequential(
            nn.Linear(history * image_feature_dim, token_dim), nn.LayerNorm(token_dim), nn.SiLU(),
        ) for key in IMAGE_KEYS})
        self.state_projection = nn.Sequential(nn.Linear(history * 8, token_dim), nn.LayerNorm(token_dim), nn.SiLU())
        self.language_projection = nn.Sequential(nn.Linear(512, token_dim), nn.LayerNorm(token_dim), nn.SiLU())
        self.sensor_identity = nn.Parameter(torch.empty(len(SENSOR_KEYS), token_dim))
        nn.init.normal_(self.sensor_identity, std=0.02)
        self.missing_tokens = nn.Parameter(torch.zeros(len(SENSOR_KEYS), token_dim))
        self.fusion = nn.Sequential(nn.Linear(len(SENSOR_KEYS) * (token_dim + 1), token_dim),
                                    nn.SiLU(), nn.Linear(token_dim, token_dim))

    def encode_tokens(self, observations: dict[str, Tensor], normalizer: Normalizer) -> Tensor:
        """Normalize uint8 on the GPU once, with no CPU float image expansion."""
        if set(observations) != set(SENSOR_KEYS):
            raise ValueError(f"Expected exactly {SENSOR_KEYS}")
        tokens = []
        for key in IMAGE_KEYS:
            images = observations[key]
            if images.dtype != torch.uint8 or images.shape[1:3] != (self.history, 3):
                raise ValueError(f"{key} must contain uint8 [B,history,3,H,W]")
            features = self.image_encoders[key](images.flatten(0, 1).float().mul_(2 / 255).sub_(1))
            tokens.append(self.temporal_projections[key](features.reshape(images.shape[0], -1)))
        tokens.append(self.state_projection(normalizer.normalize("state", observations["state"]).flatten(1)))
        tokens.append(self.language_projection(observations["language"][:, 0]))
        return torch.stack(tokens, dim=1) + self.sensor_identity.unsqueeze(0)

    def fuse_tokens(self, tokens: Tensor, presence: Tensor) -> Tensor:
        """Apply removal before cross-sensor mixing, including explicit presence flags."""
        masked = torch.where(presence[..., None], tokens, self.missing_tokens.unsqueeze(0))
        return self.fusion(torch.cat((masked.flatten(1), presence.to(tokens.dtype)), dim=-1))


@dataclass(frozen=True)
class DenoisingSample:
    """Noise, timestep and valid target shared across counterfactual sensor masks."""

    noisy_action: Tensor
    noise: Tensor
    timesteps: Tensor
    target: Tensor
    loss_mask: Tensor


class FullSensorPolicy(nn.Module):
    """Same DP backbone and widths, adapted only to two cameras and eight joint actions."""

    def __init__(self, config: dict, encoder: SensorEncoder | None = None) -> None:
        super().__init__()
        self.config = copy.deepcopy(config)
        self.horizon = config["horizon"]
        self.n_obs_steps = config["n_obs_steps"]
        self.n_action_steps = config["n_action_steps"]
        if self.horizon < self.n_obs_steps + self.n_action_steps - 1 or self.horizon % (2 ** (len(config["down_dims"]) - 1)):
            raise ValueError("Invalid action horizon")
        self.normalizer = Normalizer()
        self.obs_encoder = encoder or SensorEncoder(self.n_obs_steps, config["token_dim"])
        self.model = ConditionalUnet1D(8, config["token_dim"], config["diffusion_step_embed_dim"], tuple(config["down_dims"]))
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=config["num_diffusion_steps"],
                                             beta_schedule="squaredcos_cap_v2", variance_type="fixed_small",
                                             clip_sample=True, prediction_type="epsilon")
        self.register_buffer("condition_presence", torch.from_numpy(presence_table()))
        self.register_buffer("condition_probabilities", torch.tensor(condition_probabilities(
            config["full_input_probability"], config["mask_weights"]), dtype=torch.float32))

    def set_normalizer(self, normalizer: Normalizer) -> None:
        """Restore fitted normalization without refitting on evaluation observations."""
        self.normalizer.load_state_dict(normalizer.state_dict(), strict=True)

    def prepare_denoising_sample(self, batch: dict, noise: Tensor | None = None,
                                 timesteps: Tensor | None = None) -> DenoisingSample:
        """Zero invalid target slots and noise them identically for paired comparisons."""
        action = batch["action"]
        mask = batch["action_loss_mask"]
        if action.shape[1:] != (self.horizon, 8) or mask.shape != action.shape:
            raise ValueError("Actions and masks must have shape [B,horizon,8]")
        if not torch.all((mask == 0) | (mask == 1)) or not torch.all(mask.sum(dim=(1, 2)) > 0):
            raise ValueError("Each sample needs a binary mask and at least one supervised action")
        trajectory = torch.where(mask.bool(), self.normalizer.normalize("action", action), 0.0)
        noise = torch.randn_like(trajectory) if noise is None else noise
        timesteps = torch.randint(self.noise_scheduler.config.num_train_timesteps, (action.shape[0],),
                                  device=action.device) if timesteps is None else timesteps
        if noise.shape != action.shape or timesteps.shape != (action.shape[0],) or timesteps.dtype != torch.long:
            raise ValueError("Invalid paired noise or timestep shape/dtype")
        if not torch.all((timesteps >= 0) & (timesteps < self.noise_scheduler.config.num_train_timesteps)):
            raise ValueError("Diffusion timestep out of range")
        return DenoisingSample(self.noise_scheduler.add_noise(trajectory, noise, timesteps), noise, timesteps, noise, mask)

    def denoising_loss_per_sample(self, sample: DenoisingSample, conditioning: Tensor) -> Tensor:
        prediction = self.model(sample.noisy_action, sample.timesteps, global_cond=conditioning)
        error = (prediction.float() - sample.target.float()).square() * sample.loss_mask
        return error.sum(dim=(1, 2)) / sample.loss_mask.sum(dim=(1, 2))

    def compute_loss_per_sample(self, batch: dict, condition_ids: Tensor | None = None,
                                denoising_sample: DenoisingSample | None = None) -> Tensor:
        """Expose unreduced paired/noise-reusable losses without forcing a second image encoding."""
        batch_size = batch["action"].shape[0]
        if condition_ids is None:
            condition_ids = torch.multinomial(self.condition_probabilities, batch_size, replacement=True) if self.training else torch.zeros(
                batch_size, device=batch["action"].device, dtype=torch.long)
        if condition_ids.shape != (batch_size,) or condition_ids.dtype != torch.long:
            raise ValueError("condition_ids must be int64 [B]")
        if not torch.all((condition_ids >= 0) & (condition_ids < len(CONDITION_NAMES))):
            raise ValueError("Unknown sensor removal condition")
        tokens = self.obs_encoder.encode_tokens(batch["obs"], self.normalizer)
        conditioning = self.obs_encoder.fuse_tokens(tokens, self.condition_presence[condition_ids])
        return self.denoising_loss_per_sample(denoising_sample or self.prepare_denoising_sample(batch), conditioning)

    def compute_loss(self, batch: dict, condition_ids: Tensor | None = None,
                     denoising_sample: DenoisingSample | None = None) -> Tensor:
        return self.compute_loss_per_sample(batch, condition_ids, denoising_sample).mean()

    @torch.no_grad()
    def paired_denoising_losses(self, batch: dict, noise: Tensor | None = None,
                                timesteps: Tensor | None = None) -> dict[str, Tensor]:
        """Evaluate all real-sensor masks with exactly the same noise, target and timestep."""
        if self.training:
            raise RuntimeError("Paired comparisons require eval mode")
        sample = self.prepare_denoising_sample(batch, noise, timesteps)
        tokens = self.obs_encoder.encode_tokens(batch["obs"], self.normalizer)
        losses = torch.stack([self.denoising_loss_per_sample(sample, self.obs_encoder.fuse_tokens(
            tokens, presence.expand(tokens.shape[0], -1))) for presence in self.condition_presence], dim=1)
        return {"losses": losses, "utility": losses[:, 1:] - losses[:, :1]}

    @torch.no_grad()
    def predict_action(self, obs_dict: dict[str, Tensor], presence: Tensor | None = None) -> dict[str, Tensor]:
        """Return eight executable actions at the current observation anchor, in joint units."""
        tokens = self.obs_encoder.encode_tokens(obs_dict, self.normalizer)
        if presence is None:
            presence = self.condition_presence[0].expand(tokens.shape[0], -1)
        conditioning = self.obs_encoder.fuse_tokens(tokens, presence)
        trajectory = torch.randn(tokens.shape[0], self.horizon, 8, device=tokens.device)
        self.noise_scheduler.set_timesteps(self.config["num_inference_steps"])
        for timestep in self.noise_scheduler.timesteps:
            prediction = self.model(trajectory, timestep, global_cond=conditioning)
            trajectory = self.noise_scheduler.step(prediction, timestep, trajectory).prev_sample
        actions = self.normalizer.unnormalize("action", trajectory)
        start = self.n_obs_steps - 1
        return {"action": actions[:, start:start + self.n_action_steps], "action_pred": actions}
