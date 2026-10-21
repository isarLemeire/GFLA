from typing import Union


import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from src.data.nips_2017_dataset import NIPS2017Dataset
from typing import Optional

# Constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_dataloader(
    batch_size: int,
    subset_size: Optional[int],
    csv_filepath: str,
    img_dir_filepath: str,
) -> DataLoader:
    """Instantiates NIPS2017 dataset and returns a PyTorch DataLoader."""

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    nips_dataset = NIPS2017Dataset(
        csv_file=csv_filepath, 
        img_dir=img_dir_filepath, 
        transform=transform
    )

    if subset_size is not None and subset_size < len(nips_dataset):
            nips_dataset = nips_dataset.get_subset(subset_size)
    
    return DataLoader(
        nips_dataset,
        batch_size=batch_size,
        shuffle=False,
    )