"""Warm-up hooks for Idea5 BASD-Reg and SCOQ."""

from mmengine.hooks import Hook

from ai4rs.registry import HOOKS


def _unwrap_bbox_head(runner):
    model = runner.model
    if hasattr(model, "module"):
        model = model.module
    return getattr(model, "bbox_head", None)


@HOOKS.register_module()
class Idea5BASDRegWarmupHook(Hook):
    """Linearly warm BASD-Reg alpha while collecting EMA from epoch zero."""

    def __init__(
        self,
        start_epoch: int = 6,
        full_epoch: int = 18,
        max_alpha: float = 0.20,
        verbose: bool = True,
    ) -> None:
        self.start_epoch = int(start_epoch)
        self.full_epoch = int(full_epoch)
        self.max_alpha = float(max_alpha)
        self.verbose = bool(verbose)
        if self.start_epoch < 0:
            raise ValueError("start_epoch must be non-negative")
        if self.full_epoch <= self.start_epoch:
            raise ValueError("full_epoch must be greater than start_epoch")
        if not 0 <= self.max_alpha < 1:
            raise ValueError("max_alpha must be in [0, 1)")

    def _alpha(self, epoch: int) -> float:
        if epoch < self.start_epoch:
            return 0.0
        if epoch >= self.full_epoch:
            return self.max_alpha
        progress = (epoch - self.start_epoch) / (self.full_epoch - self.start_epoch)
        return self.max_alpha * progress

    def before_train_epoch(self, runner) -> None:
        bbox_head = _unwrap_bbox_head(runner)
        if bbox_head is None or not hasattr(bbox_head, "set_basd_reg_alpha"):
            return
        applied = bbox_head.set_basd_reg_alpha(self._alpha(runner.epoch))
        if self.verbose:
            runner.logger.info(f"Idea5 BASD-Reg alpha={applied:.4f}")


@HOOKS.register_module()
class SCOQWarmupHook(Hook):
    """Enable SCOQ loss at a fixed epoch without breaking the DDP graph."""

    def __init__(self, start_epoch: int = 6, verbose: bool = True) -> None:
        self.start_epoch = int(start_epoch)
        self.verbose = bool(verbose)
        if self.start_epoch < 0:
            raise ValueError("start_epoch must be non-negative")

    def before_train_epoch(self, runner) -> None:
        bbox_head = _unwrap_bbox_head(runner)
        if bbox_head is None or not hasattr(bbox_head, "set_scoq_loss_scale"):
            return
        requested = 1.0 if runner.epoch >= self.start_epoch else 0.0
        applied = bbox_head.set_scoq_loss_scale(requested)
        if self.verbose:
            runner.logger.info(f"SCOQ loss scale={applied:.1f}")


__all__ = ["Idea5BASDRegWarmupHook", "SCOQWarmupHook"]
