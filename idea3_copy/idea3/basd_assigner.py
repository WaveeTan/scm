from typing import List, Optional, Union

import torch
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData
from mmdet.models.task_modules.assigners.assign_result import AssignResult
from mmdet.models.task_modules.assigners.base_assigner import BaseAssigner
from scipy.optimize import linear_sum_assignment
from torch import Tensor

from ai4rs.registry import TASK_UTILS


@TASK_UTILS.register_module()
class BASDHungarianAssigner(BaseAssigner):
    """Hungarian assigner that retains the detached total cost matrix.

    BASD uses the same matrix for matching and for measuring the normalized
    entropy of the lowest-cost queries for every GT. Retaining it here avoids
    evaluating all matching costs a second time in the detection head.
    """

    def __init__(
        self,
        match_costs: Union[List[Union[dict, ConfigDict]], dict, ConfigDict],
    ) -> None:
        if isinstance(match_costs, dict):
            match_costs = [match_costs]
        if not isinstance(match_costs, list) or not match_costs:
            raise ValueError('match_costs must be a non-empty list or dict.')
        self.match_costs = [TASK_UTILS.build(cost) for cost in match_costs]

    @staticmethod
    def _attach_cost(result: AssignResult, cost: Tensor) -> AssignResult:
        # It is consumed immediately by BASDRotatedRTDETRHead and is never
        # checkpointed. Prefer AssignResult's public extension API when present.
        if hasattr(result, 'set_extra_property'):
            result.set_extra_property('basd_cost_matrix', cost)
        else:
            result.basd_cost_matrix = cost
        return result

    def assign(self,
               pred_instances: InstanceData,
               gt_instances: InstanceData,
               img_meta: Optional[dict] = None,
               **kwargs) -> AssignResult:
        num_gts = len(gt_instances)
        num_preds = len(pred_instances)
        device = pred_instances.bboxes.device
        gt_labels = gt_instances.labels

        assigned_gt_inds = torch.full((num_preds, ),
                                      -1,
                                      dtype=torch.long,
                                      device=device)
        assigned_labels = torch.full((num_preds, ),
                                     -1,
                                     dtype=torch.long,
                                     device=device)

        if num_gts == 0 or num_preds == 0:
            if num_gts == 0:
                assigned_gt_inds[:] = 0
            result = AssignResult(
                num_gts=num_gts,
                gt_inds=assigned_gt_inds,
                max_overlaps=None,
                labels=assigned_labels)
            empty_cost = pred_instances.bboxes.new_zeros((num_preds, num_gts))
            return self._attach_cost(result, empty_cost)

        cost_list = [
            match_cost(
                pred_instances=pred_instances,
                gt_instances=gt_instances,
                img_meta=img_meta) for match_cost in self.match_costs
        ]
        cost = torch.stack(cost_list).sum(dim=0).detach()
        cost = torch.nan_to_num(cost, nan=1e8, posinf=1e8, neginf=-1e8)

        matched_rows, matched_cols = linear_sum_assignment(cost.cpu().numpy())
        matched_rows = torch.as_tensor(
            matched_rows, dtype=torch.long, device=device)
        matched_cols = torch.as_tensor(
            matched_cols, dtype=torch.long, device=device)

        assigned_gt_inds[:] = 0
        assigned_gt_inds[matched_rows] = matched_cols + 1
        assigned_labels[matched_rows] = gt_labels[matched_cols]
        result = AssignResult(
            num_gts=num_gts,
            gt_inds=assigned_gt_inds,
            max_overlaps=None,
            labels=assigned_labels)
        return self._attach_cost(result, cost)


__all__ = ['BASDHungarianAssigner']
