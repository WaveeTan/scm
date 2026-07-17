from mmengine.hooks import Hook

from ai4rs.registry import HOOKS


@HOOKS.register_module()
class SceneConditionWarmupHook(Hook):
    """Linearly warm up scene bias for query selection."""

    def __init__(self,
                 start_epoch: int = 0,
                 full_epoch: int = 24,
                 min_mix: float = 0.05,
                 verbose: bool = True) -> None:
        self.start_epoch = int(start_epoch)
        self.full_epoch = int(full_epoch)
        self.min_mix = float(min_mix)
        self.verbose = bool(verbose)
        if self.full_epoch <= self.start_epoch:
            raise ValueError('full_epoch must be greater than start_epoch.')
        if not 0.0 <= self.min_mix <= 1.0:
            raise ValueError('min_mix must be in [0, 1].')

    def _get_mix(self, epoch: int) -> float:
        if epoch < self.start_epoch:
            return self.min_mix
        if epoch >= self.full_epoch:
            return 1.0
        progress = (epoch - self.start_epoch) / (
            self.full_epoch - self.start_epoch)
        mix = self.min_mix + (1.0 - self.min_mix) * progress
        return max(0.0, min(1.0, float(mix)))

    def before_train_epoch(self, runner) -> None:
        mix = self._get_mix(runner.epoch)
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module
        if not hasattr(model, 'scene_context'):
            return
        model.scene_context.set_scene_mix(mix)
        if self.verbose:
            runner.logger.info(f'Scene condition mix={mix:.4f}')
