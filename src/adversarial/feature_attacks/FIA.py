import torch
import torch.nn.functional as F
from .feature_attack import FeatureAttack
from typing import List, Tuple

class FIA(FeatureAttack):
    """
     Feature Importance-aware Attack (FIA)

    Paper:
        Title: Feature Importance-aware Transferable Adversarial Attacks
        Authors: Zhibo Wang, Hengchang Guo, Zhifei Zhang, Wenxin Liu, Zhan Qin, and Kui Ren
        Link: https://openaccess.thecvf.com/content_ICCV_2019/html/Ganeshan_FDA_Feature_Disruptive_Attack_ICCV_2019_paper.html
    """
    def __init__(
            self, 
            *args, 
            drop_rate: float=0.3, 
            num_ensemble: int=5, 
            num_classes: int = 1000, 
            attack_name: str="FIA", **kwargs
        ) -> None:
        super().__init__(*args, attack_name=attack_name, **kwargs)
        self.drop_rate = drop_rate
        self.num_ensemble = num_ensemble
        self.num_classes = num_classes

    def setup_references(self, images: torch.Tensor, labels: torch.Tensor) -> List[torch.Tensor]: # type: ignore
        """Pre-computes the aggregate gradient-based importance weights (W)."""
        importance_weights = []
        
        one_hot_labels = F.one_hot(labels, self.num_classes).float().to(self.device)

        self.model.zero_grad()

        agg_grads = None
        for _ in range(self.num_ensemble):
            # Apply random drop mask
            keep_mask = torch.bernoulli(
                torch.full_like(images, 1.0 - self.drop_rate)
            )
            x_masked = (images * keep_mask).detach()

            logits, feats = self._forward_with_features(x_masked)

            loss = (logits * one_hot_labels).sum()
            layer_grads = torch.autograd.grad(
                loss,
                feats,
                retain_graph=False,
                create_graph=False,
            )

            if agg_grads is None:
                agg_grads = [g.detach().clone() for g in layer_grads]
            else:
                for i, g in enumerate(layer_grads):
                    agg_grads[i] += g.detach()
        
        # Normalize
        normed: list[torch.Tensor] = []
        for g in agg_grads: # type: ignore
            l2 = g.norm(p=2)
            normed.append(-(g / (l2 + 1e-12)))

        importance_weights = normed
                
        return importance_weights

    def compute_loss(
        self,
        adv_images: torch.Tensor,
        labels: torch.Tensor,
        references: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, int]:
        """Calculates FDA loss"""   
        model_layer_loss = 0

        current_feats = self._get_features(adv_images)
        W_list = references
        num_layers = len(current_feats)

        if num_layers == 0:
            raise RuntimeError("Empty layer_registry")
        
        for feat, w in zip(current_feats, W_list):
            n_elements = feat.numel()
            if n_elements > 0:
                model_layer_loss += (feat * w).sum() / n_elements
        
        loss = (model_layer_loss / (num_layers))
                
        return loss, 1 # type: ignore