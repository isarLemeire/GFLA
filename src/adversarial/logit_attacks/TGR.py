import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from contextlib import contextmanager

from typing import Tuple, Optional, List
from functools import partial

from ..base_attack import BaseAttack


class TGR(BaseAttack):
    """
    Token Gradient Regularization (TGR)

    Paper:
        Title: Transferable Adversarial Attacks on Vision Transformers with Token Gradient Regularization
        Authors: Jianping Zhang, Yizhan Huang, Weibin Wu, and Michael R. Lyu
        Link: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10205153
        GitHub: https://github.com/jpzhang1810/TGR
    """

    def __init__(
        self,
        model: nn.Module,
        eps: float = 8 / 255,
        alpha: float = 2 / 255,
        steps: int = 10,
        decay: float = 0.9,
        attn_gamma: float = 0.25,
        qkv_gamma: float = 0.75,
        mlp_gamma: float = 0.5,
        attack_name: str = "TGR",
    ) -> None:
        super().__init__(model=model, eps=eps, alpha=alpha, steps=steps, attack_name=attack_name)
        self.decay = decay
        self.attn_gamma = attn_gamma
        self.qkv_gamma = qkv_gamma
        self.mlp_gamma = mlp_gamma

        self._hook_handles = []

    def _attn_tgr(
        self,
        module: nn.Module,
        grad_in: Tuple[torch.Tensor, ...],
        grad_out: Tuple[torch.Tensor, ...],
        gamma: float,
    ) -> Tuple[torch.Tensor, ...]:
        """Backward hook modifying attention gradients by zeroing out extreme token elements."""
        out_grad = grad_in[0].clone() * gamma
        B, C, H, W = out_grad.shape  # C = num_heads, H = W = num_tokens
        out_grad_cpu = out_grad.data.clone().cpu().numpy().reshape(B, C, H * W)

        max_all = np.argmax(out_grad_cpu[0, :, :], axis=1)
        max_all_H, max_all_W = max_all // H, max_all % H
        min_all = np.argmin(out_grad_cpu[0, :, :], axis=1)
        min_all_H, min_all_W = min_all // H, min_all % H

        out_grad[:, range(C), max_all_H, :] = 0.0
        out_grad[:, range(C), :, max_all_W] = 0.0
        out_grad[:, range(C), min_all_H, :] = 0.0
        out_grad[:, range(C), :, min_all_W] = 0.0
        return (out_grad,)

    def _qkv_tgr(
        self,
        module: nn.Module,
        grad_in: Tuple[torch.Tensor, ...],
        grad_out: Tuple[torch.Tensor, ...],
        gamma: float,
    ) -> Tuple[torch.Tensor, ...]:
        """Backward hook modifying QKV projection gradients by zeroing out per-channel extremes."""
        out_grad = grad_in[0].clone() * gamma
        c = out_grad.shape[2]
        out_grad_cpu = out_grad.data.clone().cpu().numpy()

        max_all = np.argmax(out_grad_cpu[0, :, :], axis=0)  # per-channel extreme token
        min_all = np.argmin(out_grad_cpu[0, :, :], axis=0)
        out_grad[:, max_all, range(c)] = 0.0
        out_grad[:, min_all, range(c)] = 0.0
        return (out_grad,) + tuple(grad_in[1:])

    def _mlp_tgr(
        self,
        module: nn.Module,
        grad_in: Tuple[torch.Tensor, ...],
        grad_out: Tuple[torch.Tensor, ...],
        gamma: float,
    ) -> Tuple[torch.Tensor, ...]:
        """Backward hook modifying MLP block gradients by zeroing out per-channel extremes."""
        out_grad = grad_in[0].clone() * gamma
        c = out_grad.shape[2]
        out_grad_cpu = out_grad.data.clone().cpu().numpy()

        max_all = np.argmax(out_grad_cpu[0, :, :], axis=0)
        min_all = np.argmin(out_grad_cpu[0, :, :], axis=0)
        out_grad[:, max_all, range(c)] = 0.0
        out_grad[:, min_all, range(c)] = 0.0
        return (out_grad,) + tuple(grad_in[1:])

    def _get_blocks(self) -> Optional[List[nn.Module]]:
        """Retrieves Transformer blocks from the target model structure."""
        if hasattr(self.model, "blocks"):
            return list(self.model.blocks)

        if hasattr(self.model, "transformers"):
            blocks = []
            for stage in self.model.transformers:
                stage_blocks = getattr(stage, "blocks", None)
                if stage_blocks is not None:
                    blocks.extend(list(stage_blocks))
            if blocks:
                return blocks

    def _register_hooks(self) -> None:
        """Registers backward hooks on target Transformer block modules."""
        attn_hook = partial(self._attn_tgr, gamma=self.attn_gamma)
        qkv_hook = partial(self._qkv_tgr, gamma=self.qkv_gamma)
        mlp_hook = partial(self._mlp_tgr, gamma=self.mlp_gamma)

        if self._get_blocks() is None: # If attacking CNNs
            return

        for block in self._get_blocks(): # type: ignore

            if hasattr(block, "attn"):
                if hasattr(block.attn, "attn_drop"):
                    self._hook_handles.append(
                        block.attn.attn_drop.register_full_backward_hook(attn_hook) # type: ignore
                    )
                if hasattr(block.attn, "qkv"):
                    self._hook_handles.append(
                        block.attn.qkv.register_full_backward_hook(qkv_hook) # type: ignore
                    )
            if hasattr(block, "mlp"):
                self._hook_handles.append(
                    block.mlp.register_full_backward_hook(mlp_hook) # type: ignore
                )

    def remove_hooks(self) -> None:
        """Removes registered gradient manipulation hooks from the model."""
        for h in self._hook_handles:
            h.remove()
        self._hook_handles = []

    @contextmanager
    def _tgr_hooks(self):
        self._register_hooks()
        try:
            yield
        finally:
            self.remove_hooks()

    # ------------------------------------------------------------------ #
    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Executes the TGR attack forward optimization loop."""
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        loss_fn = nn.CrossEntropyLoss()

        momentum = torch.zeros_like(images).detach()
        adv_images = images.clone().detach()

        with self._tgr_hooks():
            for _ in range(self.steps):
                adv_images.requires_grad_(True)
                outputs = self.get_logits(adv_images)
                cost = loss_fn(outputs, labels)

                grad = torch.autograd.grad(cost, adv_images, retain_graph=False, create_graph=False)[0]

                grad = grad / (torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True) + 1e-8)
                grad = grad + momentum * self.decay
                momentum = grad

                adv_images = adv_images.detach() + self.alpha * grad.sign()
                delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
                adv_images = torch.clamp(images + delta, min=0.0, max=1.0).detach()

        return adv_images