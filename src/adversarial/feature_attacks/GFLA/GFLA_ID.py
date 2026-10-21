import torch
import torch.nn.functional as F

from .GFLA import GFLA

from typing import Any, Dict, List, Tuple

class GFLA_ID(GFLA):
    """
    Base class for applying input diversity to the GFLA.
    The desired transformation for input diversity can be incoorperated by
    overriding _transform.
    """

    def __init__(
        self,
        *args,
        num_copies: int = 5,
        N: int = 3,
        attack_name: str = "GFLA-ID",
        **kwargs,
    ) -> None:
        super().__init__(*args, N=N, attack_name=attack_name, **kwargs)
        self.num_copies = num_copies

    def _transform(self, x: torch.Tensor, i: int) -> torch.Tensor:
        """Per-copy transform. Subclasses override this (i is the copy index)."""
        raise NotImplementedError

    def _get_neighbors(self, images: torch.Tensor) -> torch.Tensor:
        """Stack of max(1, N) noisy neighbors, shape (n*B, ...)."""
        return torch.cat(
            [self._get_neighbor(images).detach() for _ in range(max(1, self.N))], dim=0
        )

    def _get_guided_direction(
        self,
        x_neighbors: torch.Tensor,
        labels: torch.Tensor,
        n: int,
        B: int,
    ) -> List[torch.Tensor]:
        """Computes feature-space loss gradients averaged across input diversity transformed neighbors."""
        logits, feats = self._forward_with_features(x_neighbors)
        loss = F.cross_entropy(logits, labels.repeat(n))
        grads = torch.autograd.grad(loss, feats)
        return [g.detach().reshape(n, B, *g.shape[1:]).mean(0) for g in grads]

    def _feature_loss(
        self,
        adv_feats: List[torch.Tensor],
        clean_feats: List[torch.Tensor],
        fixed_grads: List[torch.Tensor],
        batch_size: int,
    ) -> torch.Tensor:
        """Calculates GFLA-ID loss"""
        layer_loss = 0.0
        num_layers = len(self.layer_registry)

        if num_layers == 0:
            raise RuntimeError("Empty layer_registry")

        for f_adv, f_clean, g in zip(adv_feats, clean_feats, fixed_grads):
            delta_f = (f_adv - f_clean).reshape(batch_size, -1)
            g_flat = g.reshape(batch_size, -1)
            g_unit = g_flat / (torch.norm(g_flat, p=2, dim=1, keepdim=True) + 1e-8)

            proj = (delta_f * g_unit).sum(dim=1)
            dist = torch.norm(delta_f, p=2, dim=1)
            layer_loss += (self.lam * proj + (1 - self.lam) * dist).sum()

        return layer_loss / (num_layers + 1e-8) # type: ignore

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Executes the GFLA-ID attack forward optimization loop."""
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        B = images.shape[0]

        momentum = torch.zeros_like(images).detach()
        adv_images = images.clone().detach()

        for _ in range(self.steps):
            grad = torch.zeros_like(adv_images)

            for i in range(self.num_copies):
                adv_images_copy = adv_images.clone().detach().requires_grad_(True)

                clean_stack  = self._get_neighbors(images)
                x_concat = torch.cat([images, clean_stack, adv_images_copy], dim=0)
                x_id = self._transform(x_concat, i)

                B_total = x_concat.shape[0]
                x_clean_id = x_id[:B]
                x_neighbors_id = x_id[B:B_total - B]
                x_adv_id = x_id[B_total - B:]

                clean_feats = self._get_features(x_clean_id)

                num_neighbors = max (1, self.N)
                fixed_grads = self._get_guided_direction(x_neighbors_id, labels, num_neighbors, B)

                _, adv_feats = self._forward_with_features(x_adv_id)
                feat_loss = self._feature_loss(adv_feats, clean_feats, fixed_grads, B)

                g_step = torch.autograd.grad(feat_loss, adv_images_copy, retain_graph=False)[0] # type: ignore
                grad += g_step

            grad = grad / float(self.num_copies)
            grad_norm = grad / (torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True) + 1e-8)
            momentum = self.momentum * momentum + grad_norm

            adv_images = adv_images.detach() + self.alpha * momentum.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0.0, max=1.0).detach()

        return adv_images

    
    