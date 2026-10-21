import torch
from .feature_attack import FeatureAttack
from typing import List, Tuple

class FDA(FeatureAttack):
    """
    Feature Disruptive Attack (FDA)

    Paper:
        Title: FDA: Feature Disruptive Attack
        Authors: Aditya Ganeshan, Vivek B.S., and R. Venkatesh Babu
        Link: https://openaccess.thecvf.com/content_ICCV_2019/html/Ganeshan_FDA_Feature_Disruptive_Attack_ICCV_2019_paper.html
    """
    def __init__(
            self, 
            *args, 
            attack_name: str="FDA", 
            **kwargs
        ) -> None:
        kwargs.setdefault('momentum', 0.0) # FDA does not use a momentum component
        super().__init__(*args, attack_name=attack_name, **kwargs)
        
        

    def compute_loss(
        self,
        adv_images: torch.Tensor,
        labels: torch.Tensor,
        references: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, int]:
        """Calculates FDA loss"""
        model_layer_loss = 0

        current_feats = self._get_features(adv_images)
        num_layers = len(current_feats)

        if num_layers == 0:
            raise RuntimeError("Empty layer_registry")
        
        for f_adv, f_clean in zip(current_feats, references):

            reduce_dims = tuple(range(1, f_clean.ndim - 1)) if f_clean.ndim == 4 else (1,)
            mu = f_clean.mean(dim=reduce_dims, keepdim=True)
            m_high = (f_clean > mu).float()
            
            dims = tuple(range(1, f_adv.ndim))
            e_high = torch.norm(f_adv * m_high, p=2, dim=dims)
            e_low = torch.norm(f_adv * (1 - m_high), p=2, dim=dims)
            
            model_layer_loss += (
                torch.log(e_low  + 1e-8)
                - torch.log(e_high + 1e-8)
            ).sum()
        
        loss = (model_layer_loss / (num_layers + 1e-8))
        
        return loss, 1 # type: ignore