"""Warmup schedule for SCNE calibration and encoder RTQD reranking."""

from mmengine.hooks import Hook

from ai4rs.registry import HOOKS


def unwrap_model(runner):
    model = runner.model
    return model.module if hasattr(model, "module") else model


@HOOKS.register_module()
class RTPScoreScheduleHook(Hook):
    """Update epoch-dependent gates without changing the DDP parameter graph."""

    def __init__(self, verbose: bool = True) -> None:
        self.verbose = bool(verbose)

    def before_train_epoch(self, runner) -> None:
        model = unwrap_model(runner)
        if not hasattr(model, "set_rtp_epoch"):
            return
        model.set_rtp_epoch(runner.epoch)
        if self.verbose:
            scene = bool(
                model.bbox_head.scene_calibration_enabled.item()
            )
            encoder = bool(model.encoder_rerank_enabled.item())
            runner.logger.info(
                "RTP-Score epoch gates: scene_calibration=%s "
                "encoder_rtqd_rerank=%s",
                scene,
                encoder,
            )

    def before_test(self, runner) -> None:
        # A standalone test run should use the trained scoring path even when
        # the checkpoint predates persistent schedule buffers.
        model = unwrap_model(runner)
        if hasattr(model, "set_rtp_epoch"):
            model.set_rtp_epoch(10**9)


__all__ = ["RTPScoreScheduleHook", "unwrap_model"]
