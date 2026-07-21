# Idea3: BASD-Reg on SCM Rotated RT-DETR

BASD-Reg keeps the original SCM/O2-RT-DETR classification, Hungarian matcher,
five-dimensional L1, KLD, auxiliary decoder, encoder, denoising and scene
losses. It changes only the reduction of matched-positive L1/KLD losses from
the final decoder layer.

## Method

Training GTs are softly assigned to six scale-density groups:

```text
tiny-sparse, tiny-dense, small-sparse, small-dense, medium, large
```

Each group tracks an EMA of final-layer rotated-IoU error. The current batch
uses the EMA from the previous update to compute:

```text
group_weight = 1 + alpha * tanh((group_error - reference_error) / temperature)
instance_weight = clip(membership @ group_weight, 0.8, 1.2)
```

The weighted L1/KLD denominator is the global sum of instance weights, keeping
the loss scale close to O2. Only after those losses have been formed does the
current batch update the EMA. The DDP reduction remains correct when positive
counts differ across ranks or a rank has no positives.

No assignment entropy, progress deficit, group softmax budget, custom matcher,
standalone angle loss, or classification reweighting remains in Idea3.

## Training-set statistics

Scale boundaries must be train-set q25/q50/q75 values, and the density boundary
must be the median density among targets below q50. For the current DIOR-R
training definition, use `train.txt + val.txt` and never `test.txt`:

```powershell
python projects/idea3/tools/compute_basd_reg_stats.py --data-root data/DIOR
```

Paste the output into the config before formal experiments. The checked-in
numbers are clearly marked bootstrap values because this workspace does not
contain `data/DIOR`.

## Training

```powershell
python tools/train.py projects/idea3/configs/idea3_scm_o2_rtdetr_r50vd_2xb4_72e_dior.py
```

The config fixes seed 42 and uses an independent work directory. For the
equivalence experiment, set `max_alpha=0.0` in `basd_reg_cfg`; all losses and
assignments should then match the SCM/O2 baseline.
