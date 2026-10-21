import torch
import torch.nn as nn
from ..base_attack import BaseAttack

class PGN(BaseAttack):
    """
    Penalizing Gradient Norm (PGN)

    Paper:
        Title: Boosting Adversarial Transferability by Achieving Flat Local Maxima
        Authors: Zhijin Ge, Hongying Liu, Xiaosen Wang, Fanhua Shang, and Yuanyuan Liu
        Link: https://proceedings.neurips.cc/paper_files/paper/2023/file/de1739eba209c682a90ec3669229ab2d-Paper-Conference.pdf
        GitHub: https://github.com/Trustworthy-AI-Group/PGN
    """

    def __init__(
            self, 
            model: nn.Module, 
            eps: float=8 / 255, 
            alpha: float=2 / 255, 
            steps: int=10, 
            decay: float=0.9, 
            beta: float=3.0, 
            gamma: float=0.5, 
            num_samples: int=5, 
            attack_name: str="PGN"
        ) ->  None:
        super().__init__(model=model, eps=eps, alpha=alpha, steps=steps, attack_name=attack_name)
        self.decay = decay
        self.num_samples = num_samples
        self.beta = beta
        self.gamma = gamma

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Executes the PGN attack forward optimization loop."""
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        momentum = torch.zeros_like(images).detach().to(self.device)
        loss_fn = nn.CrossEntropyLoss()
        adv_images = images.clone().detach()

        for _ in range(self.steps):
            g_flat = torch.zeros_like(adv_images)

            for _ in range(self.num_samples):

                noise_rand = torch.zeros_like(adv_images).uniform_(
                                    -self.eps * self.beta, self.eps * self.beta
                                )
                x1 = (adv_images + noise_rand).detach().requires_grad_(True)

                output1 = self.get_logits(x1)
                ce_loss1 = loss_fn(output1, labels)
                g1 = torch.autograd.grad(
                    ce_loss1, x1, retain_graph=False, create_graph=False
                )[0]
                g1_norm = g1 / (torch.mean(torch.abs(g1), dim=(1, 2, 3), keepdim=True) + 1e-8)

                x_star = x1 - self.alpha * g1_norm
                output_star = self.get_logits(x_star)
                ce_loss_star = loss_fn(output_star, labels)
                g_star = torch.autograd.grad(
                    ce_loss_star, x_star, retain_graph=False, create_graph=False
                )[0]

                g_flat += ((1-self.gamma)*g1 + self.gamma*g_star)

            g_flat = g_flat / self.num_samples

            g_flat_norm = g_flat / (torch.mean(torch.abs(g_flat), dim=(1, 2, 3), keepdim=True) + 1e-8)
            momentum = self.decay * momentum + g_flat_norm

            adv_images = adv_images.detach() + self.alpha * momentum.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0.0, max=1.0).detach()

        return adv_images