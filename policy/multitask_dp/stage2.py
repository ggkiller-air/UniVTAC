"""UniVTAC port of AdaViP-main's T=2 utility-conditioned Transformer policy."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .model import DenoisingSample, FullSensorPolicy, SensorEncoder
from .schema import CONDITION_NAMES, IMAGE_KEYS, SENSOR_KEYS


class FilmHyperNet(nn.Module):
    """Predict identity-initialized channel modulation for one independent image branch."""

    def __init__(self, token_dim: int = 256) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 2 * token_dim))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, token: Tensor, context: Tensor) -> Tensor:
        """Apply x'=(1+delta_gamma)*x+beta."""
        delta_gamma, beta = self.network(context).chunk(2, dim=-1)
        return (1 + delta_gamma) * token + beta


class Stage2Policy(FullSensorPolicy):
    """Six tokens, eight signed utility dimensions, four FiLMs and fresh 8D-action DP."""

    def __init__(self, config: dict, utility_rms: Tensor | None = None,
                 encoder: SensorEncoder | None = None) -> None:
        if config["n_obs_steps"] != 2 or config["token_dim"] != 256:
            raise ValueError("Stage 2 uses the inherited T=2, 256D token protocol")
        super().__init__(config, encoder)
        dimensions = len(CONDITION_NAMES) - 1
        rms = torch.ones(dimensions) if utility_rms is None else torch.as_tensor(utility_rms, dtype=torch.float32)
        if rms.shape != (dimensions,) or not torch.isfinite(rms).all() or (rms <= 0).any():
            raise ValueError("Utility RMS must contain eight finite positive values")
        self.register_buffer("utility_rms", rms.clone())
        self.utility_predictor = nn.Sequential(nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, dimensions))
        self.z_projection = nn.Sequential(nn.Linear(256, 64), nn.SiLU())
        self.utility_projection = nn.Sequential(nn.Linear(dimensions, 64), nn.SiLU())
        self.film_hypernets = nn.ModuleList([FilmHyperNet() for _ in IMAGE_KEYS])
        self.query_projection = nn.Sequential(nn.Linear(dimensions, 256), nn.SiLU(), nn.Linear(256, 256))
        nn.init.zeros_(self.query_projection[-1].weight)
        nn.init.zeros_(self.query_projection[-1].bias)
        self.cross_attention = nn.MultiheadAttention(256, 4, batch_first=True)
        self.fusion_norm = nn.LayerNorm(256)
        self.fusion_ff = nn.Sequential(nn.Linear(256, 512), nn.SiLU(), nn.Linear(512, 256))

    def initialize_from_teacher(self, teacher: FullSensorPolicy) -> None:
        """Copy only E400 EMA encoder and normalizer; keep all new DP parameters fresh."""
        for key in ("horizon", "n_obs_steps", "n_action_steps", "token_dim", "down_dims",
                    "diffusion_step_embed_dim", "num_diffusion_steps", "num_inference_steps"):
            if teacher.config[key] != self.config[key]:
                raise ValueError(f"Teacher and Stage 2 differ in {key}")
        self.obs_encoder.load_state_dict(teacher.obs_encoder.state_dict(), strict=True)
        self.set_normalizer(teacher.normalizer)

    def encode_stage2(self, observations: dict[str, Tensor], utility_override: Tensor | None = None) -> dict[str, Tensor]:
        """Apply the same detached utility, FiLM context, query and direct attention as main."""
        tokens = self.obs_encoder.encode_tokens(observations, self.normalizer)
        presence = self.condition_presence[0].expand(len(tokens), -1)
        z_t = self.obs_encoder.fuse_tokens(tokens, presence)
        u_hat = self.utility_predictor(z_t.detach())
        action_utility = u_hat.detach() if utility_override is None else utility_override.detach()
        if action_utility.shape != u_hat.shape:
            raise ValueError("Utility override must have shape [B,8]")
        context = torch.cat((self.z_projection(z_t), self.utility_projection(action_utility)), dim=-1)
        image_count = len(IMAGE_KEYS)
        modulated = torch.stack(
            [self.film_hypernets[i](tokens[:, i], context) for i in range(image_count)]
            + [tokens[:, i] for i in range(image_count, len(SENSOR_KEYS))], dim=1,
        )
        query = z_t + self.query_projection(action_utility)
        attended, _ = self.cross_attention(query[:, None], modulated, modulated, need_weights=False)
        fused = self.fusion_norm(attended[:, 0])
        conditioning = self.fusion_norm(fused + self.fusion_ff(fused))
        return {"tokens": tokens, "z_t": z_t, "u_hat": u_hat,
                "modulated_tokens": modulated, "conditioning": conditioning}

    def training_losses(self, batch: dict, stage: str,
                        denoising_sample: DenoisingSample | None = None) -> dict[str, Tensor]:
        """Isolate utility regression from both action gradients and encoder gradients."""
        if stage not in ("A", "B", "C"):
            raise ValueError("Unknown Stage 2 phase")
        encoded = self.encode_stage2(batch["obs"])
        target = batch["utility"].to(encoded["u_hat"].dtype) / self.utility_rms
        if target.shape != encoded["u_hat"].shape or not torch.isfinite(target).all():
            raise ValueError("Expected finite raw signed labels [B,8]")
        utility_loss = nn.functional.smooth_l1_loss(encoded["u_hat"], target, beta=1.0)
        action_loss = utility_loss.detach().new_zeros(()) if stage == "A" else self.denoising_loss_per_sample(
            denoising_sample or self.prepare_denoising_sample(batch), encoded["conditioning"]).mean()
        return {**encoded, "utility_target": target, "action_loss": action_loss, "utility_loss": utility_loss}

    def compute_loss_per_sample(self, batch: dict, condition_ids: Tensor | None = None,
                                denoising_sample: DenoisingSample | None = None) -> Tensor:
        """Ensure the generic loss API evaluates the actual Stage 2 action path."""
        if condition_ids is not None and (condition_ids.shape != (len(batch["action"]),)
                                         or condition_ids.dtype != torch.long or bool(condition_ids.any())):
            raise ValueError("Stage 2 is trained with full sensors only")
        encoded = self.encode_stage2(batch["obs"])
        return self.denoising_loss_per_sample(denoising_sample or self.prepare_denoising_sample(batch),
                                              encoded["conditioning"])

    def paired_denoising_losses(self, *args: object, **kwargs: object) -> dict[str, Tensor]:
        """Reject accidentally generating teacher labels through the inherited Stage 0 path."""
        raise RuntimeError("Generate counterfactual utility labels with the frozen E400 teacher")

    @torch.no_grad()
    def predict_action(self, obs_dict: dict[str, Tensor], presence: Tensor | None = None) -> dict[str, Tensor]:
        """Use the Transformer conditioning for the original joint-position DDPM rollout."""
        if presence is not None and (presence.shape != (len(obs_dict["state"]), len(SENSOR_KEYS))
                                     or not bool(presence.all())):
            raise ValueError("Stage 2 inference expects all six sensors")
        conditioning = self.encode_stage2(obs_dict)["conditioning"]
        trajectory = torch.randn(len(conditioning), self.horizon, 8,
                                 device=conditioning.device, dtype=conditioning.dtype)
        self.noise_scheduler.set_timesteps(self.config["num_inference_steps"])
        for timestep in self.noise_scheduler.timesteps:
            prediction = self.model(trajectory, timestep, global_cond=conditioning)
            trajectory = self.noise_scheduler.step(prediction, timestep, trajectory).prev_sample
        actions = self.normalizer.unnormalize("action", trajectory)
        start = self.n_obs_steps - 1
        return {"action": actions[:, start:start + self.n_action_steps], "action_pred": actions}


def stage_for_epoch(epoch: int) -> str:
    """Match main's one-based epoch phases: 1, 2–6, 7–600."""
    if epoch < 1:
        raise ValueError("Epoch is one-based")
    return "A" if epoch == 1 else "B" if epoch <= 6 else "C"


def set_stage_trainability(policy: Stage2Policy, stage: str) -> None:
    """Train utility only in A, action additions in B, and unfreeze encoder in C."""
    if stage not in ("A", "B", "C"):
        raise ValueError(stage)
    policy.obs_encoder.requires_grad_(stage == "C")
    for module in (policy.z_projection, policy.utility_projection, policy.film_hypernets,
                   policy.query_projection, policy.cross_attention, policy.fusion_norm,
                   policy.fusion_ff, policy.model):
        module.requires_grad_(stage != "A")
    policy.utility_predictor.requires_grad_(True)
    policy.normalizer.requires_grad_(False)


def make_optimizer_scheduler(policy: Stage2Policy, steps_per_epoch: int,
                             epochs: int = 600) -> tuple[torch.optim.Optimizer, object]:
    """Match main's three LR groups, 500-step warmup and encoder unfreeze ramp."""
    encoder, utility = list(policy.obs_encoder.parameters()), list(policy.utility_predictor.parameters())
    taken = {id(parameter) for parameter in encoder + utility}
    action = [parameter for parameter in policy.parameters() if id(parameter) not in taken]
    optimizer = torch.optim.AdamW([
        {"params": encoder, "lr": 8e-5, "name": "encoder"},
        {"params": utility, "lr": 1e-4, "name": "utility"},
        {"params": action, "lr": 1e-4, "name": "action"},
    ], betas=(0.95, 0.999), eps=1e-8, weight_decay=1e-6)
    total, c_start = steps_per_epoch * epochs, steps_per_epoch * 6

    def shared(step: int) -> float:
        if step < 500:
            return max(1e-3, (step + 1) / 500)
        progress = min(1.0, (step - 500) / max(1, total - 500))
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    def encoder_schedule(step: int) -> float:
        if step < c_start:
            return 0.0
        return shared(step) * min(1.0, (step - c_start + 1) / 500)

    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, (encoder_schedule, shared, shared))
