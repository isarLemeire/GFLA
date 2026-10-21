"""
Adversarial Attack Benchmarking.
"""

import argparse
import json
import time
from pathlib import Path
from typing import  List, Type

import pandas as pd
import torch
from torch.utils.data import DataLoader


# Custom Module Imports
from src.utils.get_dataloader import get_dataloader
from src.utils.get_models import get_models
from src.utils.save_df import save_df
from src.utils.get_name import model_name, attack_name
from src.utils.get_attacks import get_attacks
from src.utils.dict_to_namespace import dict_to_namespace

from src.adversarial.adversarial_generator import AdversarialGenerator
from src.models.model_wrapper import ModelWrapper

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
    """Executes the benchmarking across all surrogate models, attacks, and targets."""
        
    all_rows = [] # Holds all the experimental results 

    for s_model in surrogate_models:
        for attack_class in attacks:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start_time = time.perf_counter()

            adv_loader = AdversarialGenerator.generate_perturbed_dataset(
                loader,
                s_model,
                attack_class,
                device=device,
                eps=epsilon,
                alpha=alpha,
                steps=steps
            )

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            wall_time = time.perf_counter() - start_time

            for t_model in target_models + [s_model]:

                results = t_model.get_fooling_and_transfer_rate(
                    s_model, loader, adv_loader
                )

                all_rows.append(
                    {
                        "Surrogate": model_name(s_model),
                        "Attack": attack_name(attack_class),
                        "Target": model_name(t_model),
                        "Fooling Rate": results["fooling_rate"] * 100,
                        "Transfer Rate": results["transfer_rate"] * 100,
                        "Wall Time (s)": wall_time,
                    }
                )

            del adv_loader
            if device.type == "cuda":
                torch.cuda.empty_cache()

    df = pd.DataFrame(all_rows)

    return df

def main(config):
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution on {device}")
    
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
        description="Adversarial Transferability Benchmarking Framework"
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