import math
from typing import Optional

from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data.dataset import random_split
from torch.utils.data.dataloader import DataLoader

import pytorch_lightning as pl


class PETS(pl.LightningDataModule):
    def __init__(self, base_path, batch_size=64, num_workers=1):
        super().__init__()
        self.base_path = base_path / 'data' / 'PETS'
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None):
        # transform
        transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        data = ImageFolder(str(self.base_path))
        train, val, test = random_split(
            data, [math.floor(len(data) * 0.7), math.ceil(len(data) * 0.1), math.ceil(len(data) * 0.2)])

        # assign to use in dataloaders
        self.train_dataset = train
        self.val_dataset = val
        self.test_dataset = test

        self.train_dataset.dataset.transform = transform['train']
        self.val_dataset.dataset.transform = transform['val']
        self.test_dataset.dataset.transform = transform['val']

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
