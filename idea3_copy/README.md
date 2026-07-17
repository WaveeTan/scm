# Idea3: BASD-Loss on SCM-RotatedRTDETR

Idea3 keeps the complete SCM-RotatedRTDETR detector, including its bounded
scene-conditioned encoder query selection and `loss_scene_cls`, and replaces
the decoder Hungarian-positive reduction with BASD-Loss.

## Implemented objective

For every GT, the normalized scale is

```text
s = sqrt(w * h) / sqrt(H * W)
```

and density uses the distance to the `K_eff`-th other GT center in normalized
image coordinates:

```text
K_eff = min(K, number_of_GT - 1)
d = log(1 + K_eff / (r_K_eff^2 + eps))
```

A single-GT image has density zero. Three smooth scale boundaries and one
smooth density boundary form a partition of unity over six groups:

```text
tiny-sparse, tiny-dense, small-sparse, small-dense, medium, large
```

The final decoder layer computes detached matched quality

```text
q = sqrt(sigmoid(correct_class_logit) * rotated_IoU)
```

and updates one checkpointable EMA per group. A circular history buffer adds a
learning-speed deficit after `progress_delta` updates. Only groups with enough
global soft mass are active; softmax over their deficits gives a fixed-sum
budget.

Each decoder layer computes ambiguity from the same total cost used by its
Hungarian assignment. Since lower cost is better, the correct distribution is

```text
pi = softmax(-topk_lowest_cost / match_temperature)
H = entropy(pi) / log(K)
a = 1 + rho * H
```

`rho` is zero for the first epochs and is enabled by
`BASDAmbiguityWarmupHook`. The final layer updates the EMA once per forward;
all earlier decoder layers share the resulting budget but retain their own
matches and ambiguity values.

The matched per-instance objective is split into:

```text
classification + 5 * XYWH-L1 + 2 * (1 - Rotated-IoU)
               + Rotated-IoU^gamma * periodic-angle-loss
```

Each component is averaged inside every soft group before applying the group
budget. Unmatched-query Varifocal loss keeps its original background
normalization. Encoder proposal and denoising losses use the configured base
losses without BASD group budgeting.

## Decisions for information absent from the screenshots

- No public paper or repository matching the exact name and formulas was found;
  the screenshots are treated as the method specification.
- Scale boundaries default to normalized 16/32/96-pixel object sizes at an
  800-pixel reference. They are configurable and should be ablated or replaced
  by training-set quantiles for another dataset.
- `density_k=3`; when fewer neighbors exist, `K_eff` is used so sparse images
  are not assigned artificially high density.
- `density_boundary=4.0` corresponds to a third-neighbor distance near 0.24 in
  normalized coordinates. It is a configurable prior, not a learned value.
- Active groups require global soft mass above `0.05`, preventing negligible
  sigmoid tails from receiving a full group budget.
- EMA momentum is `0.99`; learning-speed lag is measured over 50 forwards.
- Assignment entropy uses the five lowest-cost queries and a negative cost sign.
- Ambiguity scaling warms from 0 at epoch 3 to 0.5 at epoch 12.
- BASD state is synchronized across distributed ranks and stored in model
  checkpoints. Its group numerators use DDP-correct global normalization.

## Layout

```text
projects/idea3/
├── configs/
│   └── idea3_scm_o2_rtdetr_r50vd_2xb4_72e_dior.py
└── idea3/
    ├── basd_assigner.py
    ├── basd_grouping.py
    ├── basd_hook.py
    ├── basd_losses.py
    ├── basd_rotated_rtdetr_head.py
    └── idea3_scm_rotated_rtdetr.py
```

Main config:

```text
projects/idea3/configs/idea3_scm_o2_rtdetr_r50vd_2xb4_72e_dior.py
```
