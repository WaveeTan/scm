from ai4rs.registry import MODELS
from projects.scm_rotated_rtdetr.scm_rotated_rtdetr import SCMRotatedRTDETR


@MODELS.register_module()
class Idea3SCMRotatedRTDETR(SCMRotatedRTDETR):
    """SCM-RotatedRTDETR detector paired with the BASD detection head."""


__all__ = ["Idea3SCMRotatedRTDETR"]
