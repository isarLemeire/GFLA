import torch
import torch.nn.functional as F
from .GFLA_ID import GFLA_ID

class GFLA_SI(GFLA_ID):
    """
        GFLA + scale invariance
    """
    def __init__(self, *args, attack_name: str="GFLA-SI", **kwargs):
        super().__init__(*args, attack_name=attack_name, **kwargs)

    def _transform(self, x, i):
        """Scales x"""
        return x * (1.0 / (2 ** i))