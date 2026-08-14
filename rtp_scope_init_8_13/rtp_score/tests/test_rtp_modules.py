"""Focused RTP-Score unit contracts (run in the full O2 environment)."""

import unittest
import importlib.util
from types import SimpleNamespace

if all(
    importlib.util.find_spec(name) is not None
    for name in ("torch", "mmcv", "mmdet")
):
    import torch
else:
    torch = None


@unittest.skipIf(torch is None, "requires the O2 torch/mmcv/mmdet environment")
class TestRTPModules(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Imports are deferred so dependency-free static checks still work.
        from projects.rtp_scope_init_8_13.rtp_score.rtqd import (
            RotatedThresholdQualityHead,
        )
        from projects.rtp_scope_init_8_13.rtp_score.scne import (
            SceneNegativeEvidenceHead,
            build_presence_targets,
            negative_evidence_bias,
        )
        from projects.rtp_scope_init_8_13.rtp_score.rsu import (
            RotatedSetUniquenessLoss,
        )

        cls.quality_type = RotatedThresholdQualityHead
        cls.scene_type = SceneNegativeEvidenceHead
        cls.build_presence_targets = staticmethod(build_presence_targets)
        cls.negative_evidence_bias = staticmethod(negative_evidence_bias)
        cls.rsu_type = RotatedSetUniquenessLoss

    def test_scne_bias_never_positive_and_clamped(self):
        logits = torch.tensor([[-20.0, 20.0, -1.0]])
        bias = self.negative_evidence_bias(
            logits,
            presence_threshold=0.2,
            min_bias=-1.5,
            enabled=True,
        )
        self.assertTrue(bool((bias <= 0).all()))
        self.assertGreaterEqual(float(bias.min()), -1.5)
        self.assertEqual(float(bias[0, 1]), 0.0)

    def test_scne_warmup_disables_calibration(self):
        logits = torch.randn(2, 4)
        bias = self.negative_evidence_bias(logits, enabled=False)
        self.assertTrue(torch.equal(bias, torch.zeros_like(logits)))

    def test_scne_empty_gt_target_and_topk_shape(self):
        memory = torch.randn(2, 50, 16, requires_grad=True)
        head = self.scene_type(
            embed_dims=16, hidden_dim=8, num_classes=4, topk_ratio=0.02
        )
        scene_logits = head(memory)
        self.assertEqual(tuple(scene_logits.shape), (2, 4))
        scene_logits.sum().backward()
        self.assertIsNotNone(head.token_score.weight.grad)
        self.assertIsNotNone(head.token_score.bias.grad)
        targets = self.build_presence_targets(
            [torch.empty(0, dtype=torch.long), torch.tensor([1, 1, 3])],
            4,
            reference=memory,
        )
        self.assertEqual(float(targets[0].sum()), 0.0)
        self.assertEqual(float(targets[1].sum()), 2.0)

    def test_rtqd_target_shape_and_threshold_order(self):
        head = self.quality_type(embed_dims=8)
        targets = head.soft_targets(torch.tensor([0.55, 0.75]))
        self.assertEqual(tuple(targets.shape), (2, 4))
        self.assertTrue(bool((targets[:, :-1] >= targets[:, 1:]).all()))

    def test_rtqd_empty_positive_safe(self):
        head = self.quality_type(embed_dims=8)
        logits = head(torch.randn(2, 3, 8))
        loss, mono, _ = head.loss(
            logits,
            [torch.empty(0, dtype=torch.long)] * 2,
            [torch.empty(0)] * 2,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(mono))

    def test_rsu_filters_and_keeps_top3_rivals(self):
        scores = torch.tensor(
            [
                [
                    [0.90, 0.10],
                    [0.80, 0.20],
                    [0.70, 0.30],
                    [0.60, 0.40],
                    [0.55, 0.45],
                    [0.10, 0.90],
                ]
            ]
        )
        record = SimpleNamespace(
            pos_inds=torch.tensor([0]),
            neg_inds=torch.tensor([1, 2, 3, 4, 5]),
            assigned_gt_inds=torch.tensor([0]),
            gt_labels=torch.tensor([0]),
            pairwise_iou=torch.tensor(
                [[1.0], [0.8], [0.7], [0.6], [0.2], [0.9]]
            ),
        )
        module = self.rsu_type(
            rival_iou_thr=0.3,
            max_rivals_per_gt=3,
            require_argmax_class_match=True,
        )
        selection = module.select(scores, [record])[0]
        self.assertEqual(selection.pairs.size(0), 3)
        self.assertFalse(bool((selection.pairs[:, 1] == 0).any()))
        self.assertNotIn(4, selection.pairs[:, 1].tolist())
        self.assertNotIn(5, selection.pairs[:, 1].tolist())

    def test_rsu_empty_rivals_safe_and_prefers_winner(self):
        record = SimpleNamespace(
            pos_inds=torch.tensor([0]),
            neg_inds=torch.tensor([], dtype=torch.long),
            assigned_gt_inds=torch.tensor([0]),
            gt_labels=torch.tensor([0]),
            pairwise_iou=torch.ones(1, 1),
        )
        module = self.rsu_type()
        scores = torch.tensor([[[0.9], [0.1]]], requires_grad=True)
        loss, diagnostics, _ = module(scores, [record])
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(diagnostics["rsu_num_rivals"]), 0.0)


if __name__ == "__main__":
    unittest.main()
