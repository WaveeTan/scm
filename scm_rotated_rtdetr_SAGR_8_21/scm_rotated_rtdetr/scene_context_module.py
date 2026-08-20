import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .utils import masked_mean, split_by_spatial_shapes


class SceneContextModule(nn.Module):
    """Soft scene context that produces bounded class/scale residuals.

    ``scene_mix`` is deliberately a plain Python float rather than a parameter
    or buffer. It is training/evaluation control state, not a learned model
    statistic, so it must not be averaged/swapped by EMA.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_classes: int = 20,
        num_levels: int = 3,
        num_scene_prototypes: int = 8,
        temperature: float = 0.1,
        scene_bias_scale: float = 0.10,
        num_scale_groups: int = 4,
        scale_bias_scale: float = 0.05,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError('temperature must be positive.')

        self.num_levels = int(num_levels)
        self.temperature = float(temperature)
        self.scene_bias_scale = float(scene_bias_scale)
        self.num_scale_groups = int(num_scale_groups)
        self.scale_bias_scale = float(scale_bias_scale)

        self.prototypes = nn.Parameter(
            torch.randn(num_scene_prototypes, embed_dims) * 0.02
        )
        self.level_attention = nn.Linear(embed_dims, 1)
        self.scene_proj = nn.Sequential(
            nn.LayerNorm(embed_dims),
            nn.Linear(embed_dims, embed_dims),
            nn.GELU(),
            nn.Linear(embed_dims, embed_dims),
        )
        self.scene_fusion = nn.Sequential(
            nn.Linear(embed_dims * 2, embed_dims),
            nn.GELU(),
            nn.Linear(embed_dims, embed_dims),
        )
        self.scene_norm = nn.LayerNorm(embed_dims)
        self.class_bias_head = nn.Sequential(
            nn.LayerNorm(embed_dims),
            nn.Linear(embed_dims, embed_dims),
            nn.GELU(),
            nn.Linear(embed_dims, num_classes),
        )

        self.scale_prior_table = nn.Parameter(
            torch.zeros(num_scene_prototypes, self.num_scale_groups)
        )

        # FIX 3: schedule state must not enter parameters/buffers/state_dict.
        # Default full strength makes standalone inference deterministic.
        self.scene_mix = 1.0

        self.init_weights()

    def init_weights(self) -> None:
        nn.init.zeros_(self.level_attention.weight)
        nn.init.zeros_(self.level_attention.bias)
        nn.init.zeros_(self.class_bias_head[-1].weight)
        nn.init.zeros_(self.class_bias_head[-1].bias)

    def set_scene_mix(self, value: float) -> None:
        value = max(0.0, min(1.0, float(value)))
        self.scene_mix = value

    def forward(
        self,
        memory: Tensor,
        spatial_shapes: Tensor,
        memory_mask: Tensor = None,
    ) -> dict:
        memory_levels, level_sizes = split_by_spatial_shapes(
            memory, spatial_shapes
        )
        if len(level_sizes) != self.num_levels:
            raise ValueError(
                f'Expected {self.num_levels} feature levels, got '
                f'{len(level_sizes)}.'
            )

        if memory_mask is None:
            mask_levels = [None] * len(level_sizes)
        else:
            if memory_mask.shape[:2] != memory.shape[:2]:
                raise RuntimeError(
                    'memory_mask must be [B,N] aligned with memory: '
                    f'mask={tuple(memory_mask.shape)}, '
                    f'memory={tuple(memory.shape)}.'
                )
            mask_levels = memory_mask.split(level_sizes, dim=1)

        level_features = [
            masked_mean(level_memory, level_mask)
            for level_memory, level_mask in zip(memory_levels, mask_levels)
        ]
        level_features = torch.stack(level_features, dim=1)

        level_logits = self.level_attention(level_features).squeeze(-1)
        level_weights = level_logits.softmax(dim=-1)
        pooled_scene = (
            level_features * level_weights.unsqueeze(-1)
        ).sum(dim=1)

        scene_specific = self.scene_proj(pooled_scene)
        scene_query = F.normalize(scene_specific, dim=-1)
        prototype_keys = F.normalize(self.prototypes, dim=-1)
        temperature = max(self.temperature, 1e-6)
        scene_logits = (
            scene_query @ prototype_keys.transpose(0, 1)
        ) / temperature
        scene_weights = scene_logits.softmax(dim=-1)
        prototype_context = scene_weights @ self.prototypes

        raw_scale_logits = scene_weights @ self.scale_prior_table
        bounded_scale_bias = (
            self.scale_bias_scale * torch.tanh(raw_scale_logits)
        )
        bounded_scale_bias = (
            bounded_scale_bias
            - bounded_scale_bias.mean(dim=-1, keepdim=True)
        )

        fusion_residual = self.scene_fusion(
            torch.cat([scene_specific, prototype_context], dim=-1)
        )
        scene_feature = self.scene_norm(
            scene_specific + fusion_residual
        )

        raw_class_logits = self.class_bias_head(scene_feature)
        bounded_class_bias = (
            self.scene_bias_scale * torch.tanh(raw_class_logits)
        )
        bounded_class_bias = (
            bounded_class_bias
            - bounded_class_bias.mean(dim=-1, keepdim=True)
        )

        # FIX 3: plain scalar, therefore EMA cannot smooth/swap it.
        mix = float(self.scene_mix)
        class_bias = mix * bounded_class_bias
        scale_bias = mix * bounded_scale_bias

        return dict(
            scene_feature=scene_feature,
            scene_weights=scene_weights,
            prototype_context=prototype_context,
            level_weights=level_weights,
            raw_class_logits=raw_class_logits,
            raw_class_bias=raw_class_logits,
            class_bias=class_bias,
            raw_scale_logits=raw_scale_logits,
            scale_bias=scale_bias,
            scale_prior_table=self.scale_prior_table,
            prototypes=self.prototypes,
        )
