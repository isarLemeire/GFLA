import numpy as np
import torch
import torchvision.transforms.functional as TF

import numpy as np
import torch
import torchvision.transforms.functional as TF

from .GFLA_ID import GFLA_ID
from typing import List


class GFLA_BSR(GFLA_ID):
    """
        GFLA + block shuffle rotate
    """

    def __init__(
            self, 
            *args, 
            num_block: int=2,
            max_angles: float=0.2,
            attack_name: str="GFLA_BSR", 
            **kwargs):
        super().__init__(*args, attack_name=attack_name, **kwargs)
        self.num_block = num_block
        self.max_angles = max_angles

    def _get_random_partition(self, length: int, num_block: int) -> List[int]:
        """Partitions `length` into `num_block` random integer segments >= 1."""
        if length < num_block:
            raise ValueError(f"Length ({length}) must be >= num_block ({num_block})")
        rem_length = length - num_block
        rand = np.random.uniform(size=num_block)
        rand_norm = np.round(rand * rem_length / rand.sum()).astype(int)
        rand_norm[rand_norm.argmax()] += rem_length - rand_norm.sum()
        return (rand_norm + 1).tolist()

    def _rotate_blocks(
        self, x: torch.Tensor, n_groups: int, group_size: int
    ) -> torch.Tensor:
        """Rotates each sample by an angle shared across all n_groups stacked
        copies of that sample index -- keeps clean/neighbor/adv paired."""
        angles_rad = torch.randn(group_size, device=x.device) * self.max_angles
        angles_rad = torch.clamp(angles_rad, -2 * self.max_angles, 2 * self.max_angles)
        angles_deg = torch.rad2deg(angles_rad.repeat(n_groups))
        rotated = [
            TF.rotate(x[i:i + 1], angle=angles_deg[i].item(),
                      interpolation=TF.InterpolationMode.BILINEAR)
            for i in range(x.shape[0])
        ]
        return torch.cat(rotated, dim=0)

    def _transform(self, x: torch.Tensor, i: int) -> torch.Tensor:
        """Block shuffle-and-rotate"""
        n_groups = max(1, self.N) + 2           # clean + N neighbors + adv
        group_size = x.shape[0] // n_groups     # original batch size B

        _, _, H, W = x.shape
        height_lengths = self._get_random_partition(H, self.num_block)
        width_lengths = self._get_random_partition(W, self.num_block)
        width_perm = np.random.permutation(self.num_block)
        height_perm = np.random.permutation(self.num_block)

        x_split_w = torch.split(x, width_lengths, dim=3)
        x_split_w_perm = [x_split_w[j] for j in width_perm]

        strips = []
        for strip in x_split_w_perm:
            strip_split_h = torch.split(strip, height_lengths, dim=2)
            blocks = [self._rotate_blocks(strip_split_h[j], n_groups, group_size)
                      for j in height_perm]
            strips.append(torch.cat(blocks, dim=2))
        return torch.cat(strips, dim=3)