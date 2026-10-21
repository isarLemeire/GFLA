import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from .feature_attack import FeatureAttack

class ILPD(FeatureAttack):
    """
    intermediate-level perturbation decay (ILPD)
    Paper:
        Title: Improving Adversarial Transferability via Intermediate-level Perturbation Decay
        Authors: Qizhang Li, Yiwen Guo, Wangmeng Zuo1, and Hao Chen
        Link: https://arxiv.org/abs/2304.13410
        GitHub: https://github.com/qizhangli/ILPD-attack
    """

    
    def __init__(self, *args, coef=0.1, sigma=0.05, N=1, attack_name="ILPD", **kwargs):
        super().__init__(*args, attack_name=attack_name, **kwargs)
        self.coef = coef
        self.sigma = sigma
        self.N = max(1, N)

    def _get_ilpd_hook(
        self, ori_ilout: torch.Tensor
    ):
        """Returns a hook that blends the current output with the static reference."""
        def hook_pd(module, input, output):
            out = output[0] if isinstance(output, tuple) else output
            blended = self.coef * out + (1.0 - self.coef) * ori_ilout

            if isinstance(output, tuple):
                return (blended,) + output[1:]
            return blended
        return hook_pd

    def _capture_layer_outputs(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Captures intermediate activations across target layers without autograd history."""
        outs = []

        def hook(module, input, output):
            outs.append((output[0] if isinstance(output, tuple) else output).detach())

        handles = [m.register_forward_hook(hook) for m in self.layer_registry]
        try:
            with torch.no_grad():
                self.get_logits(x)
        finally:
            for h in handles:
                h.remove()
        return outs

    def setup_references(self, images, labels): # type: ignore
        """Uses clean images as references"""
        return images.detach()

    def compute_loss(
        self,
        adv_images: torch.Tensor,
        labels: torch.Tensor,
        references: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, int]:
        """Calculates ILPD loss"""
        num_layers = len(self.layer_registry)

        if num_layers == 0:
            raise RuntimeError("Empty layer_registry")

        accumulated_grad = torch.zeros_like(adv_images)

        for _ in range(self.N):
            noisy_input = references + self.sigma * torch.randn_like(references) # type: ignore
            ori_ilouts = self._capture_layer_outputs(noisy_input)

            for orig, mod in zip(ori_ilouts, self.layer_registry):
                handle = mod.register_forward_hook(self._get_ilpd_hook(orig))
                try:
                    loss_i = F.cross_entropy(self.get_logits(adv_images), labels)
                    grad_i = torch.autograd.grad(loss_i, adv_images)[0]
                    accumulated_grad += grad_i / (self.N * num_layers)
                finally:
                    handle.remove()

        return (adv_images * accumulated_grad.detach()).sum(), 1