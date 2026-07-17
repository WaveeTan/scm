import math
from typing import Dict, List, Optional, Tuple

import torch
from mmengine.structures import InstanceData
from mmdet.utils import InstanceList, reduce_mean
from torch import Tensor

from ai4rs.registry import MODELS
from ai4rs.structures.bbox import rbbox_overlaps
from projects.rotated_rtdetr.rotated_rtdetr import RotatedRTDETRHead
from projects.rotated_rtdetr.rotated_rtdetr.varifocal_loss import VarifocalLoss

from .basd_grouping import BASDGroupState


@MODELS.register_module()
class BASDRotatedRTDETRHead(RotatedRTDETRHead):
    """Rotated RT-DETR head with budgeted adaptive scale-density loss.

    BASD is applied to Hungarian positives from every decoder layer. The final
    layer updates the online group quality exactly once, and its group budget is
    reused by the auxiliary decoder layers. Background classification, encoder
    proposal losses and denoising losses retain their standard reduction paths.
    """

    def __init__(self,
                 *args,
                 loss_angle: Optional[dict] = None,
                 basd_cfg: Optional[dict] = None,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if not isinstance(self.loss_cls, VarifocalLoss):
            raise TypeError('BASD currently requires a VarifocalLoss variant.')
        if loss_angle is None:
            raise ValueError(
                'BASD requires an explicit per-instance angle loss.')
        self.loss_angle = MODELS.build(loss_angle)

        cfg = {} if basd_cfg is None else dict(basd_cfg)
        self.match_topk = int(cfg.pop('match_topk', 5))
        self.match_temperature = float(cfg.pop('match_temperature', 1.0))
        self.angle_iou_gamma = float(cfg.pop('angle_iou_gamma', 2.0))
        initial_rho = float(cfg.pop('ambiguity_rho', 0.0))
        if self.match_topk < 2:
            raise ValueError('match_topk must be at least two for entropy.')
        if self.match_temperature <= 0:
            raise ValueError('match_temperature must be positive.')
        if self.angle_iou_gamma < 0:
            raise ValueError('angle_iou_gamma must be non-negative.')
        self.basd_state = BASDGroupState(**cfg)
        self.register_buffer('basd_ambiguity_rho',
                             torch.tensor(initial_rho, dtype=torch.float))

    def set_basd_rho(self, value: float) -> None:
        self.basd_ambiguity_rho.fill_(max(float(value), 0.0))

    def _assignment_entropy(self, cost: Tensor,
                            assigned_gt_inds: Tensor) -> Tensor:
        num_queries = cost.size(0)
        num_positive = assigned_gt_inds.numel()
        if num_positive == 0:
            return cost.new_zeros((0, ))
        topk = min(self.match_topk, num_queries)
        if topk <= 1:
            return cost.new_zeros((num_positive, ))
        gt_columns = cost[:, assigned_gt_inds]
        best_costs = torch.topk(
            gt_columns, k=topk, dim=0, largest=False).values
        probabilities = torch.softmax(
            -best_costs / self.match_temperature, dim=0)
        entropy = -(probabilities *
                    probabilities.clamp_min(1e-12).log()).sum(dim=0)
        return (entropy / math.log(topk)).clamp(0, 1).detach()

    def _get_basd_targets_single(self, cls_score: Tensor, bbox_pred: Tensor,
                                 gt_instances: InstanceData,
                                 img_meta: dict) -> Dict[str, Tensor]:
        img_h, img_w = img_meta['img_shape'][:2]
        factor = bbox_pred.new_tensor(
            [img_w, img_h, img_w, img_h, self.angle_factor]).unsqueeze(0)
        num_queries = bbox_pred.size(0)

        if hasattr(gt_instances.bboxes, 'regularize_boxes'):
            gt_instances.bboxes.regularize_boxes(**self.angle_cfg)
        gt_bboxes = gt_instances.bboxes
        if hasattr(gt_bboxes, 'tensor'):
            gt_bboxes = gt_bboxes.tensor
        gt_labels = gt_instances.labels

        pred_instances = InstanceData(
            scores=cls_score, bboxes=bbox_pred * factor)
        assign_result = self.assigner.assign(
            pred_instances=pred_instances,
            gt_instances=gt_instances,
            img_meta=img_meta)
        if hasattr(assign_result, 'get_extra_property'):
            cost_matrix = assign_result.get_extra_property('basd_cost_matrix')
        else:
            cost_matrix = getattr(assign_result, 'basd_cost_matrix', None)
        if cost_matrix is None:
            raise RuntimeError(
                'BASDRotatedRTDETRHead requires BASDHungarianAssigner.')

        pos_inds = torch.nonzero(
            assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(
            assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        pos_gt_inds = assign_result.gt_inds[pos_inds] - 1

        labels = gt_labels.new_full((num_queries, ), self.num_classes)
        bbox_targets = bbox_pred.new_zeros((num_queries, 5))
        membership = bbox_pred.new_zeros(
            (num_queries, self.basd_state.num_groups))
        ambiguity = bbox_pred.new_zeros((num_queries, ))

        if pos_inds.numel() > 0:
            labels[pos_inds] = gt_labels[pos_gt_inds]
            bbox_targets[pos_inds] = gt_bboxes[pos_gt_inds] / factor
            gt_membership = self.basd_state.compute_membership(
                gt_bboxes, img_meta['img_shape'])
            membership[pos_inds] = gt_membership[pos_gt_inds].to(
                dtype=bbox_pred.dtype)
            ambiguity[pos_inds] = self._assignment_entropy(
                cost_matrix, pos_gt_inds).to(dtype=bbox_pred.dtype)

        return dict(
            labels=labels,
            bbox_targets=bbox_targets,
            membership=membership,
            ambiguity=ambiguity,
            pos_inds=pos_inds,
            neg_inds=neg_inds,
            factor=factor.repeat(num_queries, 1))

    def _build_basd_targets(self, cls_scores: Tensor, bbox_preds: Tensor,
                            batch_gt_instances: InstanceList,
                            batch_img_metas: List[dict]) -> Dict[str, Tensor]:
        records = [
            self._get_basd_targets_single(cls_score, bbox_pred, gt_instances,
                                          img_meta)
            for cls_score, bbox_pred, gt_instances, img_meta in zip(
                cls_scores, bbox_preds, batch_gt_instances, batch_img_metas)
        ]
        num_queries = bbox_preds.size(1)
        device = bbox_preds.device
        pos_mask_list = []
        neg_mask_list = []
        for record in records:
            pos_mask = torch.zeros(
                num_queries, dtype=torch.bool, device=device)
            neg_mask = torch.zeros(
                num_queries, dtype=torch.bool, device=device)
            pos_mask[record['pos_inds']] = True
            neg_mask[record['neg_inds']] = True
            pos_mask_list.append(pos_mask)
            neg_mask_list.append(neg_mask)
        return dict(
            labels=torch.cat([record['labels'] for record in records]),
            bbox_targets=torch.cat(
                [record['bbox_targets'] for record in records]),
            membership=torch.cat([record['membership'] for record in records]),
            ambiguity=torch.cat([record['ambiguity'] for record in records]),
            factors=torch.cat([record['factor'] for record in records]),
            pos_mask=torch.cat(pos_mask_list),
            neg_mask=torch.cat(neg_mask_list),
            num_pos=sum(record['pos_inds'].numel() for record in records),
            num_neg=sum(record['neg_inds'].numel() for record in records))

    @staticmethod
    def _as_per_instance(loss: Tensor, num_instances: int) -> Tensor:
        if loss.ndim == 1:
            return loss
        return loss.reshape(num_instances, -1).sum(dim=-1)

    def _group_reduce(self,
                      values: Tensor,
                      membership: Tensor,
                      ambiguity: Tensor,
                      budget: Tensor,
                      gate: Optional[Tensor] = None) -> Tensor:
        instance_weight = 1 + self.basd_ambiguity_rho.to(
            values) * ambiguity.detach()
        soft_weight = membership.detach() * instance_weight[:, None]
        numerator_values = values if gate is None else values * gate.detach()
        local_numerator = (soft_weight * numerator_values[:, None]).sum(dim=0)
        global_denominator = self.basd_state.global_sum(
            soft_weight.sum(dim=0).detach())
        group_loss = local_numerator * self.basd_state.world_size()
        group_loss = group_loss / global_denominator.clamp_min(
            self.basd_state.eps)
        return (budget.to(group_loss) * group_loss).sum()

    def _loss_matching_single(
        self,
        cls_scores: Tensor,
        bbox_preds: Tensor,
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        update_state: bool,
        budget: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Dict[str, Tensor]]:
        targets = self._build_basd_targets(cls_scores, bbox_preds,
                                           batch_gt_instances, batch_img_metas)
        labels = targets['labels']
        pos_mask = targets['pos_mask']
        neg_mask = targets['neg_mask']
        bbox_targets = targets['bbox_targets']
        factors = targets['factors']
        membership = targets['membership'][pos_mask]
        ambiguity = targets['ambiguity'][pos_mask]

        flat_cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        flat_bbox_preds = bbox_preds.reshape(-1, 5)
        decoded_bboxes = flat_bbox_preds * factors
        decoded_targets = bbox_targets * factors

        aligned_iou = flat_bbox_preds.new_zeros(flat_bbox_preds.size(0))
        if pos_mask.any():
            aligned_iou[pos_mask] = rbbox_overlaps(
                decoded_bboxes[pos_mask],
                decoded_targets[pos_mask],
                mode='iou',
                is_aligned=True).clamp(0, 1).detach()

        cls_targets = flat_cls_scores.new_zeros(flat_cls_scores.shape)
        if pos_mask.any():
            pos_labels = labels[pos_mask]
            cls_targets[pos_mask, pos_labels] = aligned_iou[pos_mask]
        per_query_cls = self.loss_cls(
            flat_cls_scores, cls_targets,
            reduction_override='none').sum(dim=-1)

        cls_avg_factor = flat_cls_scores.new_tensor(targets['num_pos'] +
                                                    targets['num_neg'] *
                                                    self.bg_cls_weight)
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(cls_avg_factor)
        cls_avg_factor = cls_avg_factor.clamp_min(1)
        loss_cls_bg = per_query_cls[neg_mask].sum() / cls_avg_factor

        num_positive = int(pos_mask.sum().item())
        if num_positive > 0:
            unit_weight = flat_bbox_preds.new_ones((num_positive, 5))
            per_bbox = self.loss_bbox(
                flat_bbox_preds[pos_mask],
                bbox_targets[pos_mask],
                unit_weight,
                reduction_override='none')
            per_bbox = self._as_per_instance(per_bbox, num_positive)
            per_iou = self.loss_iou(
                decoded_bboxes[pos_mask],
                decoded_targets[pos_mask],
                unit_weight,
                reduction_override='none')
            per_iou = self._as_per_instance(per_iou, num_positive)
            per_angle = self.loss_angle(
                decoded_bboxes[pos_mask],
                decoded_targets[pos_mask],
                unit_weight,
                reduction_override='none')
            per_angle = self._as_per_instance(per_angle, num_positive)
        else:
            empty = flat_bbox_preds[pos_mask, 0]
            per_bbox = empty
            per_iou = empty
            per_angle = empty

        pos_cls_prob = per_query_cls.new_zeros((num_positive, ))
        if num_positive > 0:
            pos_labels = labels[pos_mask]
            pos_cls_prob = flat_cls_scores[pos_mask,
                                           pos_labels].sigmoid().detach()
        pos_iou = aligned_iou[pos_mask]
        quality = torch.sqrt((pos_cls_prob * pos_iou).clamp_min(0))

        state_info = {}
        if update_state:
            state_info = self.basd_state.update(quality, membership)
            ambiguity_sum = self.basd_state.global_sum(
                ambiguity.detach().float().sum())
            ambiguity_count = self.basd_state.global_sum(
                ambiguity.new_tensor(float(ambiguity.numel())).float())
            state_info['ambiguity_mean'] = ambiguity_sum / \
                ambiguity_count.clamp_min(1)
            budget = state_info['budget']
        if budget is None:
            budget = self.basd_state.last_budget.detach().clone()

        loss_cls_pos = self._group_reduce(per_query_cls[pos_mask], membership,
                                          ambiguity, budget)
        loss_bbox = self._group_reduce(per_bbox, membership, ambiguity, budget)
        loss_iou = self._group_reduce(per_iou, membership, ambiguity, budget)
        angle_gate = pos_iou.pow(self.angle_iou_gamma)
        loss_angle = self._group_reduce(
            per_angle, membership, ambiguity, budget, gate=angle_gate)
        loss_cls = loss_cls_pos + loss_cls_bg
        return (loss_cls, loss_bbox, loss_iou, loss_angle, budget.detach(),
                state_info)

    def loss_by_feat(self,
                     all_layers_cls_scores: Tensor,
                     all_layers_bbox_preds: Tensor,
                     enc_cls_scores: Tensor,
                     enc_bbox_preds: Tensor,
                     batch_gt_instances: InstanceList,
                     batch_img_metas: List[dict],
                     dn_meta: Optional[Dict[str, int]],
                     batch_gt_instances_ignore=None) -> Dict[str, Tensor]:
        if batch_gt_instances_ignore is not None:
            raise AssertionError('BASD does not support ignored GT instances.')
        (matching_cls, matching_bbox, dn_cls,
         dn_bbox) = self.split_outputs(all_layers_cls_scores,
                                       all_layers_bbox_preds, dn_meta)

        final = self._loss_matching_single(
            matching_cls[-1],
            matching_bbox[-1],
            batch_gt_instances,
            batch_img_metas,
            update_state=True)
        final_cls, final_bbox, final_iou, final_angle, budget, state_info = final
        loss_dict = dict(
            loss_cls=final_cls,
            loss_bbox=final_bbox,
            loss_iou=final_iou,
            loss_angle=final_angle)

        for layer_id in range(len(matching_cls) - 1):
            layer = self._loss_matching_single(
                matching_cls[layer_id],
                matching_bbox[layer_id],
                batch_gt_instances,
                batch_img_metas,
                update_state=False,
                budget=budget)
            loss_dict[f'd{layer_id}.loss_cls'] = layer[0]
            loss_dict[f'd{layer_id}.loss_bbox'] = layer[1]
            loss_dict[f'd{layer_id}.loss_iou'] = layer[2]
            loss_dict[f'd{layer_id}.loss_angle'] = layer[3]

        if enc_cls_scores is not None:
            enc_loss_cls, enc_loss_bbox, enc_loss_iou = \
                super().loss_by_feat_single(
                    enc_cls_scores,
                    enc_bbox_preds,
                    batch_gt_instances=batch_gt_instances,
                    batch_img_metas=batch_img_metas)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_loss_bbox
            loss_dict['enc_loss_iou'] = enc_loss_iou

        if dn_cls is not None:
            dn_losses_cls, dn_losses_bbox, dn_losses_iou = self.loss_dn(
                dn_cls,
                dn_bbox,
                batch_gt_instances=batch_gt_instances,
                batch_img_metas=batch_img_metas,
                dn_meta=dn_meta)
            loss_dict['dn_loss_cls'] = dn_losses_cls[-1]
            loss_dict['dn_loss_bbox'] = dn_losses_bbox[-1]
            loss_dict['dn_loss_iou'] = dn_losses_iou[-1]
            for layer_id, (loss_cls, loss_bbox, loss_iou) in enumerate(
                    zip(dn_losses_cls[:-1], dn_losses_bbox[:-1],
                        dn_losses_iou[:-1])):
                loss_dict[f'd{layer_id}.dn_loss_cls'] = loss_cls
                loss_dict[f'd{layer_id}.dn_loss_bbox'] = loss_bbox
                loss_dict[f'd{layer_id}.dn_loss_iou'] = loss_iou

        if state_info:
            for group_id, group_name in enumerate(self.basd_state.GROUP_NAMES):
                loss_dict[f'basd_budget_{group_name}'] = budget[group_id]
                loss_dict[f'basd_quality_{group_name}'] = state_info[
                    'quality'][group_id]
                loss_dict[f'basd_mass_{group_name}'] = state_info['mass'][
                    group_id]
                loss_dict[f'basd_difficulty_{group_name}'] = state_info[
                    'difficulty'][group_id]
                loss_dict[f'basd_progress_{group_name}'] = state_info[
                    'progress'][group_id]
            loss_dict['basd_ambiguity_mean'] = state_info['ambiguity_mean']
        loss_dict['basd_ambiguity_rho'] = self.basd_ambiguity_rho.detach()
        return loss_dict


__all__ = ['BASDRotatedRTDETRHead']
