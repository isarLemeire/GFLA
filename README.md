

# GFLA: Guided Feature-Level Attack

## Abstract
Deep neural networks (DNNs) are vulnerable to adversarial examples (AEs), which are test samples with carefully crafted, imperceptible perturbations that induce incorrect model predictions. In gray-box and black-box attack scenarios, AEs can be trained on a surrogate whitebox model such that the attack is transferable to unseen models. Feature-level or feature-space attacks enhance transferability by perturbing intermediate feature representations, leveraging the fact that disparate models often converge toward shared, task-specific features. To effectively cross decision boundaries, feature perturbations must be large in magnitude and be adversarially aligned. Within the input space, the gradient of the classification loss is commonly used to identify adversarial directions. Within the feature space, this directional heuristic remains under explored. Instead, existing methods utilize indirect directional heuristics to craft perturbations, reducing black-box attack efficacy. By contrast, we propose the Guided Feature-Level Attack (GFLA), a framework that explicitly maximizes feature displacement along adversarial gradients. The resulting feature disruptions are both large and directionally aligned with the surrogate model’s loss landscape, thereby increasing transferability. Extensive evaluations across diverse model architectures demonstrate that our approach consistently outperforms existing methods, establishing a new state of the art in adversarial transferability.


## Setup & Assets

### Requirements

Install the dependencies:

```bash
pip install -r requirements.txt
```

### Dataset

The experiments use the **NIPS 2017 Adversarial Attacks and Defenses** dataset.

1.  Download the dataset from [Kaggle](https://www.kaggle.com/datasets/google-brain/nips-2017-adversarial-learning-development-set).
    
2.  Save and extract the files into a root directory named `NIPS2017/`:`
```
NIPS2017/
├── images/
├── categories.csv
└── images.csv
```
### Models

Download the desired pretrained models, and save them to `./models`. Models can be downloaded as follows:

Python

```python
# 1. Standard PyTorch Vision Models
from torchvision import models
resnet50 = models.resnet50(weights='IMAGENET1K_V1')

# 2. PyTorch Image Models (timm)
import timm
inception_v3 = timm.create_model('inception_v3', pretrained=True)

# 3. RobustBench Models
from robustbench.utils import load_model
adv_vit_b = load_model(
    model_name="Mo2022When_ViT-B", 
    dataset="imagenet", 
    threat_model="Linf",
    model_dir="./models"
)
```


## Overview

This repository contains four main execution scripts. Detailed configuration examples for each script are provided in their corresponding `.json` files. The current `.json` files assume the required surrogate and target models and dataset are installed at the correct location:

* **`main.py`** – Compares the transferability and fooling rates of various adversarial attacks across specified surrogate and target models (see `main.json` or `debug.json`).
* **`ablation.py`** – Conducts hyperparameter sweeps for GFLA (see `ablation.json`).
* **`analysis.py`** – Analyzes feature perturbations induced by different attack methods (see `analysis.json`).
* **`analysis_per_iteration.py`** – Compares step-by-step optimization performance between GFLA and GFLA-D across iterations (see `analysis_per_iteration.json`).

### Quick start
Assuming the setup was performed successfully, the code can be executed using:
```bash
python main.py --config configs/main.json
```

### Supported Attacks

The repository supports **25 adversarial attack methods**:

* **Logit-level:** `BIM`, `MIFGSM`, `NIFGSM`, `DIFGSM`, `TIFGSM`, `SINIFGSM`, `VMIFGSM`, `ANDA`, `BSR`, `GRA`, `LPAA`, `MIG`, `PGN`, `TGR`
* **Feature & Intermediate Layer Attacks:** `FDA`, `FIA`, `ILA`, `ILPD`, `NAA`, `TAP`
* **GFLA Variations (Ours):** `GFLA`, `GFLA_D`, `GFLA_BSR`, `GFLA_DI`, `GFLA_SI`

Of these attacks, `BIM`, `MIFGSM`, `NIFGSM`, `DIFGSM`, `TIFGSM`, `SINIFGSM`, `VMIFGSM` are imported from `torchattacks`, the others were implemented from scratch, to be compatible with `torchattacks`.


## Repository Structure


```text
├── main.py                     	# Main attack evaluation pipeline
├── ablation.py                 	# Hyperparameter sweeps for GFLA
├── analysis.py                 	# Feature perturbation analysis
├── analysis_per_iteration.py   	# Iterative optimization analysis
│
├── configs/                    	# Experiment configuration files
│   ├── main.json
│   ├── debug.json
│   ├── ablation.json
│   ├── analysis.json
│   └── analysis_per_iteration.json
│
├── NIPS2017/                   	# Dataset (created on setup)
│   ├── images/
│
├── models/                     	# Cached models (created on setup)
│
└──  src/
	├── adversarial/				# Adversarial attacks
	│   ├── logit_attacks/			# Logit-level attacks
	│   └── feature_attacks/		# Feature-level attacks
	│		    └── GFLA/			# GFLA and variants
	├── data/						# Useful loader for NIPS2017
	├── models/						# Useful wrapper models
	└── utils/						# Useful wrapper models
```


## License

This project is released under the [MIT License](https://www.google.com/search?q=LICENSE).
