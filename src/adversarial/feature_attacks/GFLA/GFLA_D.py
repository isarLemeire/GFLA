import torch
import torch.nn.functional as F
from .GFLA import GFLA

from typing import List, Tuple

class GFLA_D(GFLA):
    def __init__(self, *args, attack_name="GFLA_D", **kwargs):
        """
        Dynamic GFLA that recalculates feature-space gradients at every step.
        """
        super().__init__(*args, attack_name=attack_name, **kwargs)


    def setup_references(self, images, labels): # type: ignore
        with torch.no_grad():
            return self._get_features(images)

    def _compute_step_gradients(self, adv_images: torch.Tensor, labels: torch.Tensor) -> List[torch.Tensor]:
        """
        Computes the current step's feature-space gradients using
        """
        step_grads = []
        num_samples = max(1, self.N)
        
        for _ in range(num_samples):
            x_neighbor = self._get_neighbor(adv_images)
            logits, feats = self._forward_with_features(x_neighbor)
            
            loss = F.cross_entropy(logits, labels)
            grads = torch.autograd.grad(loss, feats)
            
            if not step_grads:
                step_grads = [g.detach() / num_samples for g in grads]
            else:
                for i, g in enumerate(grads):
                    step_grads[i] += g.detach() / num_samples
                
        return step_grads

    def compute_loss(
            self,
            adv_images: torch.Tensor,
            labels: torch.Tensor,
            references: List[torch.Tensor],
        ) -> Tuple[torch.Tensor, int]:
        """Calculates GFLA-D loss"""
        batch_size = adv_images.shape[0]

        current_step_grads = self._compute_step_gradients(adv_images, labels)

        feats = self._get_features(adv_images)
        
        model_layer_loss = 0
        num_layers = len(self.layer_registry)

        if num_layers == 0:
            raise RuntimeError("Empty layer_registry")
        
        for f_adv, f_clean, g in zip(
            feats, 
            references, 
            current_step_grads):
            
            delta_f = (f_adv - f_clean).reshape(batch_size, -1)
            g_flat = g.reshape(batch_size, -1)

            g_norm = torch.norm(g_flat, p=2, dim=1, keepdim=True) + 1e-8
            g_unit = g_flat / g_norm

            # Guided component
            proj = (delta_f * g_unit).sum(dim=1)
            # Unguided component
            dist = torch.norm(delta_f, p=2, dim=1)

            model_layer_loss += (self.lam * proj + (1 - self.lam) * dist).sum()

        loss = (model_layer_loss / (num_layers + 1e-8))
        return loss, 1 # type: ignore