# RTP-Score for O2-RT-DETR

This independent project implements the scoring decomposition from
`RTP_Score_O2_RT_DETR_Agent_Project_Spec`. New RTP-Score code lives under
`projects/rtp_scope`; the original O2-RT-DETR implementation is only used as
the inherited baseline.

## What is implemented

- **Rotated-IACS fix**: rotated/probabilistic Varifocal targets use the scale
  factor of each image in a variable-size batch.
- **SCNE**: encoder mean plus learned top-k token pooling, asymmetric
  image-level presence loss, warmup, and strictly non-positive class bias.
  The R50 DIOR experiment uses LayerNorm and four independent evidence pools,
  followed by a `512 -> 256` MLP, so different scene patterns do not have to
  compete for one shared salient-token set.
- **RTQD**: final and optional encoder heads predict the probability of
  crossing rotated IoU thresholds `(0.5, 0.6, 0.7, 0.8)`. Only Hungarian
  positives are supervised; unmatched queries are ignored.
- **Encoder reranking**: classification preselects up to 900 tokens, then a
  warmed RTQD/classification geometric score selects 300 decoder queries.
- **RSU**: final matching-query winner/rival ranking with at most three
  same-class, rIoU-gated rivals per GT. It reuses the final-layer Hungarian
  assignment used by the standard detector loss.
- **Optional uniqueness graph head**: available behind
  `rsu.use_unique_head`, but disabled in the main configs.
- **NMS-free prediction**: SCNE calibration -> class probability -> q50 ->
  optional uniqueness -> log-space fusion -> flattened top-k -> OBB decode.
- **Diagnostics**: query dump, AP50 false-positive taxonomy, score/rIoU
  calibration, and oracle reranking under `tools/analysis_tools/rtp_score`.

SCNE logits are explicitly passed through encoder output, pre-decoder and
head inputs. No mutable detector cache is used. Schedule gates are persistent
buffers and are DDP/checkpoint safe.

## Configs

| Config | Purpose |
|---|---|
| `_base_rtp_score.py` | Parity base; RTP ranking modules off |
| `rtp_r18_dior_rotated_iacs.py` | Rotated-IoU Varifocal target |
| `rtp_r18_dior_rtqd_final.py` | Final decoder RTQD |
| `rtp_r18_dior_rtqd_encoder.py` | Final + encoder RTQD |
| `rtp_r18_dior_rsu.py` | Pairwise RSU |
| `rtp_r18_dior_scne.py` | Negative-only SCNE |
| `rtp_r18_dior_all.py` | Joint R18 DIOR-R experiment |
| `rtp_r50_dior_all.py` | Joint R50 DIOR-R experiment; 4-head enhanced SCNE |
| `rtp_r18_dota_all.py` | Joint DOTA experiment |
| `rtp_r34_fair1m_all.py` | Joint FAIR1M experiment |
| `rtp_r18_dronevehicle_rgb_all.py` | DOTA-format RGB DroneVehicle template |

The DroneVehicle integration is a template because the source tree has no
native DroneVehicle dataset class. It expects five-class DOTA text annotations
under `data/split_ss_dronevehicle/`; adjust class spellings and paths to match
the local conversion.

## Offline gate

Enable query export with:

```python
model["test_cfg"]["export_query_details"] = True
```

Then run:

```powershell
python tools/analysis_tools/rtp_score/dump_query_predictions.py `
  --input work_dirs/rtp/query_results.pkl `
  --output work_dirs/rtp/query_dump.json
python tools/analysis_tools/rtp_score/analyze_ap50_errors.py `
  --dump work_dirs/rtp/query_dump.json `
  --output work_dirs/rtp/ap50_errors.json
python tools/analysis_tools/rtp_score/oracle_rerank.py `
  --dump work_dirs/rtp/query_dump.json `
  --output work_dirs/rtp/oracle.json
```

## Training

R18 RTQD-final:

```powershell
python tools/train.py projects/rtp_scope/configs/rtp_r18_dior_rtqd_final.py
```

R50 joint model:

```powershell
python tools/train.py projects/rtp_scope/configs/rtp_r50_dior_all.py
```

The schedule hook trains the SCNE auxiliary loss from epoch 0 but enables
negative calibration at epoch 8. Encoder RTQD is trained from the start and
begins affecting query selection at epoch 8.

Main experiments intentionally reject `test_cfg.nms`. Use a separate
diagnostic config for a rotated-NMS upper bound.
