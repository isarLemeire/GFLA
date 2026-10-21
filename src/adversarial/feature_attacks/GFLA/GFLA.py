import torch
import torch.nn.functional as F
from ..feature_attack import FeatureAttack

from typing import Any, Dict, List, Tuple

class GFLA(FeatureAttack):
    """
    Guided feature-level Attack (GFLA)

    The GFLA maximizes two components:
        - The guided component: maximizes the feature perturbation along the direction which maxizes the CE-loss
        - The unguided component: maximizes the feature perturbation magnitude
    """

    def __init__(
        self,
        *args: Any,
        lam: float = 0.9,
        N: int = 5,
        beta: float = 4.5,
        attack_name: str = "GFLA",
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, attack_name=attack_name, **kwargs)
        self.lam = lam
        self.N = N  # Neighborhood samples
        self.beta = beta # Noise scale factor

    def _get_neighbor(self, images: torch.Tensor) -> torch.Tensor:
        """Generates a noisy input sample from the uniform epsilon-neighborhood."""
        images_det = images.detach()
        if self.N > 0:
            noise = (torch.rand_like(images_det) * 2 - 1) * (self.eps * self.beta)
            return (images_det + noise).clamp(0, 1).requires_grad_(True)
        return images_det.clone().requires_grad_(True)

    def setup_references(  # type: ignore
        self, images: torch.Tensor, labels: torch.Tensor
    ) -> Dict[str, List[torch.Tensor]]:
        """
        Pre-computes clean features and averaged feature-space gradients on the clean input.
        """
        clean_feats = self._get_features(images)

        num_samples = max(1, self.N)  
        accumulated_grads = [torch.zeros_like(f) for f in clean_feats]

        for _ in range(num_samples):
            x_neighbor = self._get_neighbor(images)

            logits, feats = self._forward_with_features(x_neighbor)
            loss = F.cross_entropy(logits, labels)

            # Compute gradients w.r.t intermediate feature maps
            grads = torch.autograd.grad(
                loss, feats, retain_graph=False, create_graph=False
            )

            for acc, g in zip(accumulated_grads, grads):
                acc.add_(g.detach())

        # Average gradients across all sampled neighbors
        fixed_grads = [acc / num_samples for acc in accumulated_grads] # type: ignore

        return {
            "clean_feats": clean_feats,
            "fixed_grads": fixed_grads,
        }

    def compute_loss(
        self,
        adv_images: torch.Tensor,
        labels: torch.Tensor,
        references: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, int]:
        """Calculates GFLA loss"""
        batch_size = adv_images.shape[0]
        clean_feats = references['clean_feats'] # type: ignore
        fixed_grads = references['fixed_grads'] # type: ignore

        feats = self._get_features(adv_images)

        layer_loss = 0
        num_layers = len(self.layer_registry)

        if num_layers == 0:
            raise RuntimeError("Empty layer_registry")

        for f_adv, f_clean, g in zip(
            feats,
            clean_feats,
            fixed_grads):
            
            delta_f = (f_adv - f_clean).reshape(batch_size, -1)
            g_flat = g.reshape(batch_size, -1)

            g_norm = torch.norm(g_flat, p=2, dim=1, keepdim=True) + 1e-8
            g_unit = g_flat / g_norm

            # Guided component
            proj = (delta_f * g_unit).sum(dim=1)
            # Unguided component
            dist = torch.norm(delta_f, p=2, dim=1)

            layer_loss += (self.lam * proj + (1 - self.lam) * dist).sum()

        loss = layer_loss / num_layers

        return loss, 1 # type: ignore