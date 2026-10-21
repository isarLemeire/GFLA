import torch
import torch.nn as nn
from ..base_attack import BaseAttack

class MIG(BaseAttack):
    """
    Momentum Integrated Gradients (MIG)

    Paper:
        Title: Transferable Adversarial Attack for Both Vision Transformers and Convolutional Networks via Momentum Integrated Gradients
        Authors: Wenshuo Ma, Yidong Li, Xiaofeng Jia3, and Wei Xu
        Link: https://ieeexplore.ieee.org/abstract/document/10378472
    """

    def __init__(
            self, 
            model: nn.Module, 
            eps: float=8 / 255, 
            alpha: float=2 / 255, 
            steps: int=10, 
            decay: float=0.9, 
            num_samples: int=5, 
            attack_name: str="MIG"
        ) -> None:
        super().__init__(model=model, eps=eps, alpha=alpha, steps=steps, attack_name=attack_name)
        self.decay = decay
        self.num_samples = num_samples

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Executes the MIG attack forward optimization loop."""
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        momentum = torch.zeros_like(images).detach().to(self.device)
        loss_fn = nn.CrossEntropyLoss()
        adv_images = images.clone().detach()

        for _ in range(self.steps):
            IG = torch.zeros_like(adv_images)

            # b = 0
            for k in range(1, self.num_samples + 1):
                mix = (k / float(self.num_samples)) * adv_images
                mix = mix.detach().requires_grad_(True)

                outputs = self.get_logits(mix)
                cost = loss_fn(outputs, labels)

                G = torch.autograd.grad(
                    cost, mix, retain_graph=False, create_graph=False
                )[0]

                IG += G

            IG = adv_images * (IG / float(self.num_samples))

            IG_norm = IG / (torch.mean(torch.abs(IG), dim=(1, 2, 3), keepdim=True) + 1e-8)
            momentum = self.decay * momentum + IG_norm

            adv_images = adv_images.detach() + self.alpha * momentum.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0.0, max=1.0).detach()

        return adv_images