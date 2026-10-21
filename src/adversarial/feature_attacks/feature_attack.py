import torch
import torch.nn as nn
from torchattacks.attack import Attack

from contextlib import contextmanager
from typing import Any, List, Tuple, Type, Union

import torch
from ..base_attack import BaseAttack

class FeatureAttack(BaseAttack):
    def __init__(
        self,
        model: nn.Module,
        eps: float = 8 / 255,
        alpha: float = 2 / 255,
        steps: int = 10,
        momentum: float = 0.9,
        layer_target_types: Tuple[Type[nn.Module], ...] = (
            nn.Conv2d,
            nn.Linear,
            nn.LayerNorm,
            nn.BatchNorm2d,
        ),
        cnn_layer_types: Tuple[Type[nn.Module], ...] = (
            nn.Conv2d,
            nn.BatchNorm2d,
        ),
        vit_layer_substrings: Tuple[str, ...] = (
            "mlp.fc2",
            "norm2",
            "attn.proj",
            "attn.qkv",
            "pool",
        ),
        attack_name: str = "FeatureAttack",
        lower_percentile: float = 0.0,
        upper_percentile: float = 1.0,
    ) -> None:
        super().__init__(model=model, eps=eps, alpha=alpha, steps=steps, attack_name=attack_name)
        self.momentum = momentum
        
        # Extraction configuration
        self.layer_target_types = layer_target_types
        self.cnn_layer_types = cnn_layer_types
        self.vit_layer_substrings = vit_layer_substrings


        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile

        self.layer_registry = self._select_layers()



    def _select_layers(self) -> List[nn.Module]:
        """
        Extracts layers based on instance-specific architectural components.
        Selection logic: (Type ∈ cnn_types OR Name ∈ vit_substrings) AND Type ∈ target_types
        """

        self.model.eval()
        candidate_modules = []       

        for n, m in self.model.named_modules():
            n_lower = n.lower()

            is_allowed_type = isinstance(m, self.layer_target_types)
            is_cnn_match = isinstance(m, self.cnn_layer_types)
            is_vit_match = any(sub in n_lower for sub in self.vit_layer_substrings)

            if is_allowed_type and (is_cnn_match or is_vit_match):
                candidate_modules.append((n, m))

        total = len(candidate_modules)
        if total == 0:
            return []
        
        if total == 1:
            return [m for (_, m) in candidate_modules]
        
        selected_modules = []
        for idx, (n, m) in enumerate(candidate_modules):
            # Compute relative depth percentile (0.0 to 1.0)
            relative_depth = idx / (total - 1)
            
            if self.lower_percentile <= relative_depth <= self.upper_percentile:
                selected_modules.append(m)
                
        return selected_modules

    @contextmanager
    def _capture(self):
        """Context manager registering forward hooks to capture feature activations."""
        outs = []
        def hook(m, i, o):
            outs.append(o[0] if isinstance(o, tuple) else o)
        handles = [m.register_forward_hook(hook) for m in self.layer_registry]
        try:
            yield outs
        finally:
            for h in handles:
                h.remove()

    def _forward_with_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Computes logits and feature activations while preserving autograd graph."""
        with self._capture() as feats:
            logits = self.get_logits(x)
        return logits, feats

    def _get_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Computes feature activations while preserving autograd graph."""
        _, feats = self._forward_with_features(x)
        return feats

    def setup_references(self, images: torch.Tensor, labels: torch.Tensor) -> List[torch.Tensor]:
        """
        Called once before the optimization loop
        Computes "reference". Each attack may use different references.
        Default computes clean feature activations"""
        with torch.no_grad():
            return self._get_features(images)

    def compute_loss(
        self,
        adv_images: torch.Tensor,
        labels: torch.Tensor,
        references: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, int]:
        """Calculates loss and optimization direction (1 for ascent, -1 for descent)."""
        raise NotImplementedError

    def forward(self, images, labels):
        """Executes the forward optimization loop, in a MIFGSM manner."""
        images = images.clone().detach().to(self.device)
        labels = labels.to(self.device)

        references = self.setup_references(images, labels)

        adv_images = images.clone().detach()
        momentum = torch.zeros_like(images).to(self.device)

        for _ in range(self.steps):
            adv_images.requires_grad = True
            
            total_loss, direction = self.compute_loss(adv_images, labels, references)
            
            grad = torch.autograd.grad(total_loss, adv_images)[0]
            grad = grad / (torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True) + 1e-8)
            momentum = self.momentum * momentum + grad

            # direction: 1 for Ascent, -1 for Descent
            adv_images = adv_images.detach() + (direction * self.alpha * momentum.sign())
            delta = torch.clamp(adv_images - images, -self.eps, self.eps)
            adv_images = torch.clamp(images + delta, 0, 1)

    
        return adv_images