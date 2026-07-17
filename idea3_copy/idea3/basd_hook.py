from mmengine.hooks import Hook

from ai4rs.registry import HOOKS


@HOOKS.register_module()
class BASDAmbiguityWarmupHook(Hook):
    """Linearly enable assignment-entropy weighting after early matching."""

    def __init__(self,
                 start_epoch: int = 3,
                 full_epoch: int = 12,
                 max_rho: float = 0.5,
                 verbose: bool = True) -> None:
        self.start_epoch = int(start_epoch)
        self.full_epoch = int(full_epoch)
        self.max_rho = float(max_rho)
        self.verbose = bool(verbose)
        if self.full_epoch <= self.start_epoch:
            raise ValueError('full_epoch must be greater than start_epoch.')
        if self.max_rho < 0:
            raise ValueError('max_rho must be non-negative.')

    def _rho(self, epoch: int) -> float:
        if epoch < self.start_epoch:
            return 0.0
        if epoch >= self.full_epoch:
            return self.max_rho
        ratio = (epoch - self.start_epoch) / (
            self.full_epoch - self.start_epoch)
        return self.max_rho * ratio

    def before_train_epoch(self, runner) -> None:
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module
        bbox_head = getattr(model, 'bbox_head', None)
        if bbox_head is None or not hasattr(bbox_head, 'set_basd_rho'):
            return
        rho = self._rho(runner.epoch)
        bbox_head.set_basd_rho(rho)
        if self.verbose:
            runner.logger.info(f'BASD ambiguity rho={rho:.4f}')


__all__ = ['BASDAmbiguityWarmupHook']
