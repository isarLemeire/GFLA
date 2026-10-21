import torch
import torch.nn.functional as F

from .GFLA_ID import GFLA_ID

class GFLA_DI(GFLA_ID):
    """
        GFLA + diverse input
    """
    def __init__(self, model, resize_rate=1.15, diversity_prob=0.5,
                attack_name="GFLA_DI", **kwargs):
        super().__init__(model=model, attack_name=attack_name, **kwargs)
        self.resize_rate = resize_rate
        self.diversity_prob = diversity_prob

    def _transform(self, x: torch.Tensor, i: int) -> torch.Tensor:
        """Applies random spatial resizing random zero-padding."""
        img_size = x.shape[-1]  # Original height/width (e.g., 224)
        img_resize = int(img_size * self.resize_rate)

        if self.resize_rate < 1:
            img_size, img_resize = img_resize, img_size

        rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
        rescaled = F.interpolate(x, size=[rnd.item(), rnd.item()], mode="nearest") # type: ignore

        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem.item() + 1, size=(1,)) # type: ignore
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem.item() + 1, size=(1,)) # type: ignore
        pad_right = w_rem - pad_left

        padded = F.pad(
            rescaled,
            [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()],
            value=0,
        )

        # Resize back to original img_size to maintain ViT input shape compatibility
        out = F.interpolate(padded, size=[img_size, img_size], mode="bilinear", align_corners=False)

        return out if torch.rand(1) < self.diversity_prob else x
    