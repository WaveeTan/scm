"""Configuration audit for reproducible RTP-Score experiments."""

from mmengine.hooks import Hook

from ai4rs.registry import HOOKS

from .rtp_schedule_hook import unwrap_model


@HOOKS.register_module()
class RTPScoreDiagnosticsHook(Hook):
    """Reject accidental NMS and print active score-factorization switches."""

    def __init__(self, require_nms_free: bool = True) -> None:
        self.require_nms_free = bool(require_nms_free)

    def before_run(self, runner) -> None:
        model = unwrap_model(runner)
        head = getattr(model, "bbox_head", None)
        if head is None or not hasattr(head, "rtp_score_cfg"):
            return
        nms_cfg = getattr(head, "test_cfg", {}).get("nms", None)
        if self.require_nms_free and nms_cfg is not None:
            raise ValueError(
                "RTP-Score main experiments are NMS-free; remove test_cfg.nms"
            )
        runner.logger.info(
            "RTP-Score active: IACS=%s SCNE=%s RTQD-final=%s "
            "RTQD-encoder=%s RSU=%s unique-head=%s",
            getattr(head, "varifocal_loss_iou_type", "unknown"),
            head.scne_enabled,
            head.final_rtqd_enabled,
            head.encoder_rtqd_enabled,
            head.rsu_enabled,
            head.unique_head_enabled,
        )


__all__ = ["RTPScoreDiagnosticsHook"]
