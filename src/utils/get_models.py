from src.models.model_wrapper import ModelWrapper
from typing import Iterable
import torch
from types import SimpleNamespace

def get_models(
        models: Iterable[SimpleNamespace], 
        device: torch.device):
    """Loads and wraps model into ModelWrapper."""
    return [
        ModelWrapper.load(
            name=getattr(model, "name", None), # type: ignore
            timm_string=getattr(model, "timm_string", None), # type: ignore
            path=getattr(model, "path", None),
            input_size=getattr(model, "input_size", 224),
            ViT=getattr(model, "ViT", False),
            device=device,
        )
        for model in models 
    ]