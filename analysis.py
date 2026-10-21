"""
Investigates feature perturbations caused by attacks
"""

import torch
import torch.nn.functional as F
from contextlib import contextmanager
import torch.nn as nn

import pandas as pd
import json
import argparse
from pathlib import Path

from typing import Any, Dict, List, Sequence, Tuple, Type, Union
from torch.utils.data import DataLoader

# Custom Module Imports
from src.utils.get_dataloader import get_dataloader
from src.utils.get_models import get_models
from src.utils.save_df import save_df
from src.utils.get_name import model_name, attack_name
from src.utils.get_attacks import get_attacks
from src.utils.dict_to_namespace import dict_to_namespace

from src.models.model_wrapper import ModelWrapper
from src.adversarial.adversarial_generator import AdversarialGenerator

def extract_model_layers(
    model: nn.Module,
    layer_target_types: Tuple[Type[nn.Module], ...] = (
        nn.Conv2d,
        nn.Linear,
        nn.LayerNorm,
        nn.BatchNorm2d,
    ),
    cnn_layer_types: Tuple[Type[nn.Module], ...] = (nn.Conv2d, nn.BatchNorm2d),
    vit_layer_substrings: Sequence[str] = (
        "mlp.fc2",
        "norm2",
        "attn.proj",
        "attn.qkv",
        "pool",
    ),
) -> List[Dict[str, nn.Module]]:
    """Filters and returns candidate target modules from a model based on layer types and Vision Transformer substring patterns."""
    candidate_modules = []
    
    for n, m in model.named_modules():
        n_lower = n.lower()
        
        is_allowed_type = isinstance(m, layer_target_types)
        is_cnn_match = isinstance(m, cnn_layer_types)
        is_vit_match = any(sub in n_lower for sub in vit_layer_substrings)

        if is_allowed_type and (is_cnn_match or is_vit_match):
            candidate_modules.append({'name': n, 'mod': m})
    
    return candidate_modules

@contextmanager
def _capture_features(modules: Sequence[nn.Module]):
    """Context manager that registers forward hooks to capture intermediate module outputs during forward passes."""
    feats = []

    def hook(module, inputs, output):
        if output.requires_grad:
            output.retain_grad()
        feats.append(output[0] if isinstance(output, tuple) else output)

    handles = [m.register_forward_hook(hook) for m in modules]
    try:
        yield feats
    finally:
        for h in handles:
            h.remove()


def _layer_metrics(
    f_clean: torch.Tensor,
    f_adv: torch.Tensor,
    grad: torch.Tensor,
    batch_size: int,
) -> Tuple[float, float, float]:
    """Computes batch-mean Relative Feature Disruption (RFD), gradient-displacement cosine similarity, and projection distance for a single layer."""
    disp = (f_adv - f_clean).reshape(batch_size, -1)
    g = grad.reshape(batch_size, -1)

    cosine = F.cosine_similarity(disp, g, dim=1)
    proj = (disp * g).sum(dim=1) / (g.norm(p=2, dim=1) + 1e-8)  # along unit gradient

    rms_disp = disp.pow(2).mean(dim=1).sqrt()
    rms_clean = f_clean.reshape(batch_size, -1).pow(2).mean(dim=1).sqrt()
    rfd = (rms_disp / (rms_clean + 1e-8)) ** 0.5

    return rfd.mean().item(), cosine.mean().item(), proj.mean().item()


def evaluate_feature_disruption(
    model: nn.Module,
    clean_loader: DataLoader,
    perturbed_loader: DataLoader,
    device: torch.device = torch.device("cuda"),
) -> Tuple[float, float, float]:
    """Evaluates relative feature disruption, cosine similarity, and projection metrics across model layers for clean versus adversarial inputs."""
    """
    Returns (RFD, cosine similarity, projection), 
    each averaged over layers, then over samples
    """
    layer_mods = [l['mod'] for l in extract_model_layers(model)]
    if not layer_mods:
        return 0.0, 0.0, 0.0

    num_layers = len(layer_mods)
    total_rfd = 0.0
    total_cosine = 0.0
    total_proj = 0.0
    num_samples = 0

    for (x_clean, y, _), (x_adv, _) in zip(clean_loader, perturbed_loader):
        B = x_clean.shape[0]
        x_clean = x_clean.to(device).requires_grad_(True)
        x_adv, y = x_adv.to(device), y.to(device)

        # Clean pass: features and the CE gradient w.r.t. each feature
        with _capture_features(layer_mods) as clean_feats:
            loss = F.cross_entropy(model(x_clean), y)
        grads = torch.autograd.grad(loss, clean_feats, allow_unused=True)
        grads = [torch.zeros_like(f) if g is None else g
                 for f, g in zip(clean_feats, grads)]
        clean_feats = [f.detach() for f in clean_feats]

        # Adversarial pass: features only
        with torch.no_grad(), _capture_features(layer_mods) as adv_feats:
            model(x_adv)

        batch_rfd = batch_cosine = batch_proj = 0.0
        for fc, fa, g in zip(clean_feats, adv_feats, grads):
            rfd, cosine, proj = _layer_metrics(fc, fa, g, B)
            batch_rfd += rfd
            batch_cosine += cosine
            batch_proj += proj

        # Layer-average, then weight the batch by its size
        total_rfd += batch_rfd / num_layers * B
        total_cosine += batch_cosine / num_layers * B
        total_proj += batch_proj / num_layers * B
        num_samples += B

    if num_samples == 0:
        return 0.0, 0.0, 0.0

    return (total_rfd / num_samples,
            total_cosine / num_samples,
            total_proj / num_samples)
    
def perform_experiment(
    attacks: List[Type],
    surrogate_models: List[ModelWrapper],
    target_models: List[ModelWrapper],
    loader: DataLoader,
    epsilon: float,
    alpha: float,
    steps: int,
    device: torch.device,
) -> pd.DataFrame:
    all_rows = []

    for s_model in surrogate_models:
        for attack_class in attacks:

            adv_loader = AdversarialGenerator.generate_perturbed_dataset(
                loader, 
                s_model, 
                attack_class,
                device=device,
                eps=epsilon,
                alpha = alpha,
                steps=steps
            )
            
            for t_model in target_models + [s_model]:                
                # 2. Get Fooling Rate
                displacement, alignment, projection = evaluate_feature_disruption(
                    model=t_model, 
                    clean_loader=loader, 
                    perturbed_loader = adv_loader, 
                    device=device) # type: ignore
                
                results = t_model.get_fooling_and_transfer_rate(s_model, loader, adv_loader)
                
                # Append data
                all_rows.append({
                    "Surrogate": model_name(s_model),
                    "Attack": attack_name,
                    "Target": model_name(t_model),
                    "Fooling Rate": results["fooling_rate"] * 100,
                    "Transfer Rate": results["transfer_rate"] * 100,
                    "CE Loss": results["loss"], 
                    "Feature disruption distance": displacement,
                    "Cosine alignment" : alignment,
                    "Feature projection": projection,
                    "Loss over displacement":  results["loss"] / displacement,
                })

            del adv_loader
            if device.type == "cuda":
                torch.cuda.empty_cache()

    return pd.DataFrame(all_rows)

def main(config):
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    
    loader = get_dataloader(
            batch_size=config.data.batch_size, 
            subset_size=getattr(config.data, "subset_size", None), 
            csv_filepath=config.data.csv_filepath, 
            img_dir_filepath=config.data.img_dir_filepath
        )
    
    surrogates = get_models(config.surrogates, device)
    targets = get_models(config.targets, device)
    
    attacks = get_attacks(config.attacks)
    
    df = perform_experiment(
        attacks=attacks,
        surrogate_models=surrogates,
        target_models=targets, 
        loader=loader,
        epsilon=config.eps,
        alpha=config.alpha,
        steps=config.steps,
        device=device
    )

    save_df(df, Path(config.output))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare the feature perturbation across different attacks"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the JSON configuration file.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)

    config = dict_to_namespace(config_dict)
        
    main(config)