import torch
import torch.nn as nn
from torchattacks.attack import Attack


class BaseAttack(Attack):
    """
    Base class for custom attacks extending torchattacks.Attack.
    """

    def __init__(
        self,
        model: nn.Module,
        eps: float = 8 / 255,
        alpha: float = 2 / 255,
        steps: int = 10,
        attack_name: str | None = None,
    ) -> None:
        name = attack_name if attack_name is not None else self.__class__.__name__
        super().__init__(name, model)

        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.supported_mode = ["default"]