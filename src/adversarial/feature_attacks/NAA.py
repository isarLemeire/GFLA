import torch
import torch.nn.functional as F
from .feature_attack import FeatureAttack
from typing import Dict, List, Tuple

class NAA(FeatureAttack):
    """
    Neuron Attribution-based Attacks (NAA)

    Paper:
        Title: Improving Adversarial Transferability via Neuron Attribution-based Attacks
        Authors: Jianping Zhang, Weibin Wu, Jen-tse Huang, Yizhan Huang, Wenxuan Wang, Yuxin Su, and Michael R. Lyu
        Link: https://ieeexplore.ieee.org/abstract/document/9879952
        GitHub: https://github.com/jpzhang1810/NAA
    """
    
    def __init__(
            self, 
            *args,
            num_steps_ig: int=5, 
            random_noise: float = 0.2, 
            attack_name: str="NAA", 
            **kwargs):
        super().__init__(*args, attack_name=attack_name, **kwargs)
        self.num_steps_ig = int(num_steps_ig)
        self.random_noise = random_noise

    def setup_references( # type: ignore
        self, images: torch.Tensor, labels: torch.Tensor
    ) -> Dict[str, List[torch.Tensor]]:
        """Computes Neuron Attribution weights (W) and Baseline Features (B)."""
        attributions = []
        baseline = torch.zeros_like(images).to(self.device)

        self.model.zero_grad()

        base_features = self._get_features(baseline)

        # 2. Integrated Gradients loop with Noise Injection
        IA = None

        for step in range(1, self.num_steps_ig + 1):
            alpha = step / self.num_steps_ig

            # interpolate new x
            noise = torch.randn_like(images) * self.random_noise # not mentioned in paper, but given in git
            x_noisy = images + noise
            x_interp = baseline + alpha * (x_noisy - baseline)
            x_interp = x_interp.detach().requires_grad_(True)
            
            logits, layer_feats = self._forward_with_features(x_interp)
            prob = F.softmax(logits, dim=1)
            true_class_prob = prob[torch.arange(labels.size(0), device=labels.device), labels].sum()

            grads = torch.autograd.grad(
                true_class_prob,
                layer_feats,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )
            
            if IA is None:
                IA = [g.detach() for g in grads]
            else:
                for i in range(len(grads)):
                    IA[i] += grads[i].detach()


        # 3. Final Weights: Negated and Normalized Average Gradient
        for g in IA: # type: ignore
            reduce_dims = tuple(range(1, g.ndim))
            norm = torch.norm(g, p=2, dim=reduce_dims, keepdim=True) + 1e-8 # not mentioned in paper, but given in git
            attributions.append(-(g / norm))
                
        return {"attributions": attributions, "baselines": base_features}

    def compute_loss(
        self,
        adv_images: torch.Tensor,
        labels: torch.Tensor,
        references: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, int]:
        """Calculates NAA loss"""  
        model_layer_loss = 0
        current_feats = self._get_features(adv_images)
        W_list = references["attributions"] # type: ignore
        B_list = references["baselines"] # type: ignore
        num_layers = len(current_feats)

        if num_layers == 0:
            raise RuntimeError("Empty layer_registry")
        
        for y_1, W, y_2 in zip(current_feats, W_list, B_list):
            # Official NAA: attribution = (adv_feat - base_feat) * weights
            # Weights are already negated, thus maximize
            A_y = (y_1 - y_2) * W
            
            # W A_y = A_y (NAA paper found the optimal values)
            # f_p and f_n are linear
            # gamma = 1
            
            # Official git does not include l_1 normalization mentioned in paper
            model_layer_loss += (A_y.sum() / y_1.numel())
        
        loss = (model_layer_loss / (num_layers + 1e-8))
                
        return loss, 1 # type: ignore