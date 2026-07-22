# Idea5：SCM + BASD-Reg + SCOQ

Idea5 在 Idea3 的 SCM + BASD-Reg 上增加 **Scene-Conditioned Oriented
Quality Calibration（SCOQ）**。它只对最终 decoder 的普通 matching queries
预测类别无关的质量分数，并在推理 top-k **之前**用几何平均融合：

```text
score = class_prob ** (1 - beta) * quality_prob ** beta
```

## 实现边界

- SCM 的 encoder top-k 场景偏置不变；
- BASD-Reg 的分组、EMA 和最终层 L1/KLD 加权不变；
- encoder、DN queries 和辅助 decoder 层不增加 quality loss；
- 最终 Hungarian 分配同时服务 BASD-Reg 与 SCOQ，不重复匹配；
- `quality_detach_inputs=True` 时，quality loss 不回传 decoder/SCM，但
  `scene_gamma` 仍可学习；
- 同一 checkpoint 下，`quality_beta=0` 直接调用原 Idea3 后处理；仍需用
  同 batch 数值测试确认 parity，不能替代独立训练的 Idea3 baseline。

默认 `quality_supervise_negatives=True` 按原方案把 unmatched query 标为 0，
因此该输出是 joint objectness-localization quality，并非纯粹的
`E[rIoU | object]`；与已有 HBB-IACS 分类分数相乘可能重复计算质量。
`quality_supervise_negatives=False, quality_loss_type="mse"` 是更接近
Cascade-DETR matched-positive IoU branch 的必做对照。

本仓库的 RT-DETR decoder 已在 decoder 内部把 256 维 query 映射成类别
logits，因此不能把 `RotatedRTDETRHead.forward` 的 `hidden_states[-1]`
当作 query 特征。Idea5 使用项目内 decoder 显式返回最终 query，避免维度
和层级错配。

## 训练

```powershell
python tools/train.py projects/idea5/configs/idea5_scoq_scm_o2_rtdetr_r50vd_2xb4_72e_dior.py
```

从 Idea3 checkpoint 做 3～6 epoch 探针时，应使用 `load_from` 而不是
`resume=True`，把 `SCOQWarmupHook.start_epoch` 改为 0，并显式冻结 backbone、
neck、encoder、decoder、原 cls/reg head 与 scene module；
`quality_detach_inputs=True` 本身**不会**冻结这些模块。正式训练前，若数据
划分变化，请重新运行：

```powershell
python projects/idea5/tools/compute_basd_reg_stats.py --data-root data/DIOR
```

## 必做对照与证伪

1. 训练前用 oracle rIoU 重排；oracle 也不能改善 AP75 时立即停止；
2. `hbox_iou` VFL（Idea3）对比仅改一行的 `rbox_iou` / `prob_iou` VFL；
3. 独立 quality head：matched-positive MSE 与原方案 all-query VFL；
4. Q1：`use_scene_conditioning=False`，纯 rotated-quality head；
5. Q2：正确 scene feature、`scene_gamma=0`、无固定点的 batch-shuffled scene；
6. 同一 checkpoint 以 `quality_beta=0/0.25/0.50` 测试；
7. 若声称协同，必须比较 SCM、SCM+BASD、SCM+SCOQ、
   SCM+BASD+SCOQ 并报告交互项；
8. 报告 AP50、AP75、score-rIoU Spearman、quality Brier/ECE，并使用多 seed。

当前继承的 SCM `val_dataloader` 使用 `test.txt`，Idea5 配置已默认关闭周期
validation。正式实验必须从现有 train+val 集合划出 held-out validation、
将其从训练集移除并重算 BASD 统计；任何 epoch、loss、target、seed 或 beta
选择都不能查看 test，协议冻结后只运行一次 test。

`DOTAMetric(iou_thrs=[0.5, 0.75])` 会分别输出 `AP50`、`AP75`，同时把
`mAP` 定义为二者均值；这个 `mAP` 既不是旧实验的 AP50，也不是 COCO
AP50:95，不能跨口径比较。

## 研究定位

普通 IoU quality head、IoU target 和几何分数融合都有强先例。可辩护的
问题是：**已有 scene representation 能否在 HBB-IACS 分类分数之上，提供
有效的 image-conditioned rotated calibration**。如果 Q2 不能稳定优于
Q1、shuffled-scene 负控和直接 `rbox_iou/prob_iou` VFL，这一方向应降级为
工程消融，而不应作为独立算法创新。

- 工程可行性：高（约 8/10），改动局限在最终 query、独立质量头与推理排序；
- 单模块原创性：低（约 2/10），IoU quality prediction 与 score fusion 已成熟；
- 系统/应用创新性：低到中（约 4/10），仅 scene-conditioned rotated
  calibration 是尚可检验的增量。

最强近邻包括 [Cascade-DETR](https://openaccess.thecvf.com/content/ICCV2023/html/Ye_Cascade-DETR_Delving_into_High-Quality_Universal_Object_Detection_ICCV_2023_paper.html)
的 final-query expected-IoU 分支、[QA-DETR](https://doi.org/10.1109/CIPCV65863.2025.00012)
针对 oriented DETR 的质量对齐 cost/loss、[AQE](https://doi.org/10.1109/TGRS.2023.3292111)
的旋转角质量估计与分数融合，以及
[PQA](https://ojs.aaai.org/index.php/AAAI/article/view/38411) 对 box-level IoU
quality 的结构性批评。因此不能声称“首次在旋转检测/DETR 中使用质量估计”，
只能围绕 scene 条件信息是否带来可证伪的额外校准收益来定位。
