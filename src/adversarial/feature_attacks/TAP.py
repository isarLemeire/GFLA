import torch
import torch.nn.functional as F
from .feature_attack import FeatureAttack
from typing import Any, Tuple, Dict

class TAP(FeatureAttack):
    """
    Transferable Adversarial Perturbations (TAP)

    Paper:
        Title: Transferable Adversarial Perturbations
        Authors: Wen Zhou, Xin Hou, Yongjun Chen, Mengyun Tang, Xiangqi Huang, Xiang Gan, and Yong Yang
        Link: https://link.springer.com/chapter/10.1007/978-3-030-01264-9_28
    """

    def __init__(
            self, 
            *args, 
            lam: float=0.005, 
            power: float= 0.5, 
            eta:float =1e3, 
            attack_name: str="TAP", 
            **kwargs
        ):
        super().__init__(*args, attack_name=attack_name, **kwargs)
        self.lam = lam
        self.power = power
        self.eta = eta

        self.kernel = torch.full((3, 3), fill_value=1.0 / 9.0, dtype=torch.float32) # 3x3 spatial domain box (averaging) linear filter

    def T(self, feat: torch.Tensor) -> torch.Tensor:
        """Applies non-linear power-scaling mapping to intermediate feature representations."""
        return torch.sign(feat) * torch.float_power(torch.abs(feat) + 1e-8, self.power)

    def compute_response_map(self, perturbation: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        """Applies a kernel filter to the perturbation."""
        B, C, H, W = perturbation.shape
        padding = kernel.shape[-1] // 2

        if kernel.dim() == 2:
            kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(C, 1, 1, 1)

        return F.conv2d(
            perturbation, 
            kernel.to(perturbation.device), 
            padding=padding, 
            groups=C
        )

    def setup_references( # type: ignore
        self, images: torch.Tensor, labels: torch.Tensor
    ) -> Dict[str, Any]:
        """Uses the clean images and the clean feature activations as references."""
        with torch.no_grad():
            return {
                'clean_images': images.clone().detach(),
                'clean_feats': self._get_features(images)
            }


    def compute_loss( # type: ignore
        self,
        adv_images: torch.Tensor,
        labels: torch.Tensor,
        references: Dict[str, Any],
    ) -> Tuple[torch.Tensor, int]:
        """Calculates TAP loss"""   
        batch_size = adv_images.shape[0]
        ce_loss = 0
        f_loss = 0

        logits, f_adv_list = self._forward_with_features(adv_images)
        ce_loss += F.cross_entropy(logits, labels)

        num_layers = len(f_adv_list)

        if num_layers == 0:
            raise RuntimeError("Empty layer_registry")

        clean_images = references['clean_images'] # type: ignore
        clean_feats = references['clean_feats'] # type: ignore
        perturbation = adv_images - clean_images

        for f_adv, f_clean in zip(f_adv_list, clean_feats):
            t_clean = self.T(f_clean).reshape(batch_size, -1)
            t_adv = self.T(f_adv).reshape(batch_size, -1)
            delta = t_clean - t_adv

            f_loss += torch.sum(delta ** 2, dim=1).mean()

        avg_f_loss = (f_loss / (num_layers + 1e-8))

        R = self.compute_response_map(perturbation, self.kernel)
        R_loss = torch.abs(R).sum()

        loss = ce_loss + self.lam * avg_f_loss + self.eta * R_loss

        return loss, 1