# SCM-RotatedRTDETR

SCM-RotatedRTDETR is a scene-conditioned ablation of O2-RTDETR.

It keeps RotatedRTDETR's original:

- global top-k encoder query selection
- encoder regression proposal path
- decoder and DN query generation
- Hungarian matcher
- detection losses

The only query-selection change is:

```python
selection_logits = enc_outputs_class + scene_outputs['class_bias'][:, None, :]
topk_indices = torch.topk(selection_logits.max(-1)[0], k=num_queries, dim=1)[1]
```

The scene branch uses learnable scene prototypes to produce a bounded class
logit residual. `SceneConditionWarmupHook` starts from a small nonzero mix and
linearly enables the bias. Detection losses and matcher stay unchanged; the
detector adds only an image-level `loss_scene_cls` term to keep the scene
branch trainable during warm-up.

Main config:

- `configs/scm_o2_rtdetr_r50vd_2xb4_72e_dior.py`
