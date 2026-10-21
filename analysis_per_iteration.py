"""
Evaluates the performance of GFLA and GFLA-D per iteration
"""

from torch.utils.data import DataLoader, TensorDataset

from pathlib import Path
import pandas as pd
import json
import argparse
from typing import List, Type, Any


from src.models.model_wrapper import ModelWrapper

from src.utils.get_dataloader import get_dataloader
from src.utils.get_models import get_models
from src.utils.save_df import save_df
from src.utils.get_name import model_name, attack_name
from src.utils.get_attacks import get_attacks
from src.utils.dict_to_namespace import dict_to_namespace

from src.adversarial.feature_attacks.GFLA.GFLA import GFLA
from src.adversarial.feature_attacks.GFLA.GFLA_D import GFLA_D

import torch
import torch.nn as nn
from tqdm import tqdm

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class TrajectoryMixin:
    """Mixin class that intercepts attack loss computation and forward passes to capture image tensors across all optimization steps."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.trajectory: List[List[torch.Tensor]] = []   # one list of tensors per forward() call (per batch)
        self._current: List[torch.Tensor] = []

    def compute_loss(self, adv_images: torch.Tensor, labels: torch.Tensor, references: Any):
        """Appends the detached CPU copy of current adversarial images during step iterations before delegating to the superclass loss function."""
        self._current.append(adv_images.detach().cpu())
        return super().compute_loss(adv_images, labels, references) # type: ignore

    def forward(self, images: torch.Tensor, labels: torch.Tensor):
        """Executes the forward attack trajectory recording intermediate states and appending the final output iterate."""
        self._current = []
        adv_images = super().forward(images, labels) # type: ignore
        self._current.append(adv_images.detach().cpu())   # final iterate
        self.trajectory.append(self._current)
        return adv_images


class GFLA_tracked(TrajectoryMixin, GFLA):
    pass


class GFLA_D_tracked(TrajectoryMixin, GFLA_D):
    pass
    

def generate_perturbed_dataset(
    loader: DataLoader, 
    model: nn.Module, 
    attack_class: type,
    device: torch.device = torch.device('cpu'),
    dataset_mean:List = IMAGENET_MEAN,
    dataset_std:List = IMAGENET_STD,
    **attack_kwargs 
) -> list[DataLoader]:
    """Generates step-wise DataLoaders representing the dataset's adversarial trajectory across all attack optimization steps."""
    
    # Prepare model
    model.eval()
    model.to(device)       
    
    # Prepare attack
    attack = attack_class(model, **attack_kwargs)
    attack.set_normalization_used(
        mean=dataset_mean,
        std=dataset_std
    )
    attack.set_device(device)

    attack.set_mode_default()
    
    step_accumulators = [[] for _ in range(attack.steps)]
    all_labels = []

    for batch in tqdm(loader, desc=f"Tracking Trajectory ({attack.attack})", leave=False):
        images = batch[0].to(device)
        labels = batch[1].to(device)

        _ = attack(images, labels)

        traj = attack.trajectory.pop() 
        # Harvest states across the entire time horizon T
        for t, x in enumerate(traj[1:]):      # skip the clean image -> S points
            step_accumulators[t].append(x)
        
        all_labels.append(labels.cpu())

    attack.trajectory.clear()
    
    trajectory_loaders = []

    flat_labels = torch.cat(all_labels)
    batch_size = loader.batch_size if loader.batch_size is not None else 32

    for t in range(attack.steps):
        step_images = torch.cat(step_accumulators[t])
        step_dataset = TensorDataset(step_images, flat_labels)
        step_loader = DataLoader(
            step_dataset, 
            batch_size=batch_size, 
            shuffle=False
        )
        trajectory_loaders.append(step_loader)
        
    return trajectory_loaders

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
    """Evaluates cross-model transferability and Cross-Entropy loss trajectories across multiple attack classes and surrogate-target model pairs at each iteration."""

    all_rows = []


    for s_model in surrogate_models:
        for attack_class in attacks:
            loaders = generate_perturbed_dataset(
                    loader, 
                    s_model, 
                    attack_class,
                    lam = 1,
                    eps=epsilon,
                    alpha = alpha,
                    steps=steps,
                    device=device
                )
            eval_loaders = [loader] + loaders
            
            for i, current_loader in tqdm(enumerate(eval_loaders)):
                for t_model in target_models + [s_model]:
                    
                    results = t_model.get_fooling_and_transfer_rate(s_model, loader, current_loader)
                    
                    all_rows.append({
                        "Surrogate": model_name(s_model),
                        "Attack": attack_name,
                        "Target": model_name(t_model),
                        "CE loss": results["loss"],
                        "Fooling Rate": results["fooling_rate"] * 100,
                        "Transfer Rate": results["transfer_rate"] * 100,
                        "step": i
                    })
                    
                    if t_model != s_model:
                        t_model.to("cpu")
                        
            del loaders
            del eval_loaders
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
    
    
    df = perform_experiment(
        attacks=[GFLA_tracked, GFLA_D_tracked],
        surrogate_models=surrogates,
        target_models=targets, 
        loader=loader,
        epsilon=config.eps,
        alpha=config.alpha,
        steps=config.steps,
        device=device # type: ignore
    )

    save_df(df, Path(config.output))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the loss at each iteration of GFLA and GFLA-D"
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