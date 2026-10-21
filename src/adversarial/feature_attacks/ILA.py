import torch
import torch.nn.functional as F
from .feature_attack import FeatureAttack
from typing import List, Tuple

class ILA(FeatureAttack):
    """
    Intermediate Level Attack (ILA)

    Paper:
        Title: Enhancing Adversarial Example Transferability With an Intermediate Level Attack
        Authors: Qian Huang, Isay Katsman, Horace He, Zeqi Gu, Serge Belongie, and Ser-Nam Lim
        Link: https://openaccess.thecvf.com/content_ICCV_2019/html/Huang_Enhancing_Adversarial_Example_Transferability_With_an_Intermediate_Level_Attack_ICCV_2019_paper.html
    """

    def __init__(
            self, 
            *args, 
            attack_name: str="ILA", 
            **kwargs):
        # ILA does not use a momentum component
        kwargs.setdefault('momentum', 0.0)
        super().__init__(*args, attack_name=attack_name, **kwargs)

    def setup_references(self, images: torch.Tensor, labels: torch.Tensor) -> List[torch.Tensor]: # type: ignore
        """Pre-computes the clean features and the features of the guide sample."""
        # 1. Generate Guide via 10-step BIM
        guide_images = images.clone().detach().requires_grad_(True)
        
        for _ in range(10):
            outputs = self.get_logits(guide_images)
            loss = F.cross_entropy(outputs, labels)
            
            # Average the loss across the ensemble
            avg_loss = loss
            grad = torch.autograd.grad(avg_loss, guide_images)[0] # type: ignore
            
            # BIM
            guide_images = (guide_images + self.alpha * grad.sign()).detach()
            delta = torch.clamp(guide_images - images, -self.eps, self.eps)
            guide_images = torch.clamp(images + delta, 0, 1).requires_grad_(True)
        
        # 2. Capture both Clean and Guide features for all models
        with torch.no_grad():
            clean_feats = self._get_features(images)
            guide_feats = self._get_features(guide_images.detach())
        
        return {"clean": clean_feats, "guide": guide_feats} # type: ignore

    def compute_loss(
        self,
        adv_images: torch.Tensor,
        labels: torch.Tensor,
        references: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, int]:
        """Calculates ILA loss"""       
        model_layer_loss = 0
        current_feats = self._get_features(adv_images)
        num_layers = len(current_feats)

        if num_layers == 0:
            raise RuntimeError("Empty layer_registry")
        
        for i, f_adv in enumerate(current_feats):
            f_clean = references["clean"][i] # type: ignore
            f_guide = references["guide"][i] # type: ignore
            
            # Project current feature perturbation onto guide perturbation
            dg = (f_guide - f_clean).reshape(f_clean.shape[0], -1)
            da = (f_adv - f_clean).reshape(f_clean.shape[0], -1)
            
            projection = (da * dg).sum(dim=1).sum()
            model_layer_loss += projection
        
        loss = (model_layer_loss / (num_layers + 1e-8))
                
        return loss, 1 # type: ignore