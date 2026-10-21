"""
Perform parameter sweeps for GFLA
"""

import argparse
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

from typing import Any, Dict, List, Sequence, Tuple, Type, Union, Optional
import pandas as pd
import torch

from src.adversarial.adversarial_generator import AdversarialGenerator
from src.models.model_wrapper import ModelWrapper
from torch.utils.data import DataLoader


from src.utils.get_dataloader import get_dataloader
from src.utils.get_models import get_models
from src.utils.save_df import save_df
from src.utils.get_name import model_name
from src.utils.dict_to_namespace import dict_to_namespace

from src.adversarial.adversarial_generator import AdversarialGenerator
from src.adversarial.feature_attacks.GFLA.GFLA import GFLA

def as_dict(ns: Optional[Union[SimpleNamespace, Dict[str, Any]]]) -> Dict[str, Any]:
    """SimpleNamespace (or dict, or None) -> plain dict."""
    if ns is None:
        return {}
    return vars(ns) if isinstance(ns, SimpleNamespace) else dict(ns)


def sweep_configs(sweep: Optional[Union[SimpleNamespace, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Cartesian product of the swept parameters -> list of {param: value} dicts."""
    sweep_dict = as_dict(sweep)
    
    names = list(sweep_dict)
    combos = [dict(zip(names, values)) for values in itertools.product(*sweep_dict.values())]

    # Percentile sweeps only make sense when lower < upper
    return [
        c for c in combos
        if not ("lower_percentile" in c and "upper_percentile" in c
                and c["lower_percentile"] >= c["upper_percentile"])
    ]


def perform_experiment(
    surrogate_models: List[ModelWrapper],
    target_models: List[ModelWrapper],
    loader: DataLoader,
    sweep: Optional[Union[SimpleNamespace, Dict[str, Any]]],
    steps: int,
    attack_kwargs: Optional[Union[SimpleNamespace, Dict[str, Any]]] = None,
    device: torch.device = torch.device("cuda"),
) -> pd.DataFrame:
    """Executes hyperparameter sweeps across surrogate and target model pairs."""

    attack_kwargs = as_dict(attack_kwargs)
    configs = sweep_configs(sweep)
    all_rows = []

    for s_model in surrogate_models:
        for params in configs:
            adv_loader = AdversarialGenerator.generate_perturbed_dataset(
                loader,
                s_model,
                GFLA,
                device=device,  # type: ignore
                **attack_kwargs,
                **params,
                steps=steps
            )

            for t_model in target_models + [s_model]:

                results = t_model.get_fooling_and_transfer_rate(
                    s_model, loader, adv_loader
                )
                all_rows.append({
                    "Surrogate": model_name(s_model),
                    "Attack": "GFLA",
                    "Target": model_name(t_model),
                    "Fooling Rate": results["fooling_rate"] * 100,
                    "Transfer Rate": results["transfer_rate"] * 100,
                    **params,
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

    df = perform_experiment(
        surrogate_models=surrogates,
        target_models=targets,
        loader=loader,
        sweep=config.sweep,
        steps=config.steps,
        attack_kwargs=config.attack_kwargs,
        device=device,
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