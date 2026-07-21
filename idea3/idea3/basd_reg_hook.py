"""Warm-up schedule for BASD-Reg adaptive strength."""

from mmengine.hooks import Hook

from ai4rs.registry import HOOKS


@HOOKS.register_module()
class BASDRegWarmupHook(Hook):
    """Linearly warm alpha while allowing EMA collection from epoch zero."""

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
        model = runner.model
        if hasattr(model, "module"):
            model = model.module
        bbox_head = getattr(model, "bbox_head", None)
        if bbox_head is None or not hasattr(bbox_head, "set_basd_reg_alpha"):
            return
        requested = self._alpha(runner.epoch)
        applied = bbox_head.set_basd_reg_alpha(requested)
        if self.verbose:
            runner.logger.info(f"BASD-Reg alpha={applied:.4f}")


__all__ = ["BASDRegWarmupHook"]
