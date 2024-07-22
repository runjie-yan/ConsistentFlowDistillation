from dataclasses import dataclass

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset

import threestudio
from threestudio import register
from threestudio.utils.base import BaseObject
from threestudio.utils.config import parse_structured
from threestudio.utils.typing import *


@dataclass
class PlaceholderDataModuleConfig:
    # height, width, and batch_size should be Union[int, List[int]]
    # but OmegaConf does not support Union of containers
    batch_size: Any = 1
    n_val_views: int = 1
    n_test_views: int = 4


class PlaceholderIterableDataset(BaseObject, IterableDataset):
    @dataclass
    class Config(PlaceholderDataModuleConfig):
        pass

    cfg: Config

    def configure(self) -> None:
        pass

    def __iter__(self):
        while True:
            yield {}

    def collate(self, batch) -> Dict[str, Any]:
        # sample elevation angles
        B = self.cfg.batch_size

        return {
            "batch_size": B,
            "elevation": torch.zeros(B, device=self.device),
            "azimuth": torch.zeros(B, device=self.device),
            "camera_distances": torch.ones(B, device=self.device),
            "c2w": torch.eye(4, device=self.device)[None].repeat(B, 1, 1),
            "mvp_mtx": None,
        }


class PlaceholderDataset(BaseObject, Dataset):
    @dataclass
    class Config(PlaceholderDataModuleConfig):
        pass

    cfg: Config

    def configure(self, split: str) -> None:
        self.split = split

        if split == "val":
            self.n_views = self.cfg.n_val_views
        else:
            self.n_views = self.cfg.n_test_views

    def __len__(self):
        return self.n_views

    def __getitem__(self, index):
        return {"index": index}

    def collate(self, batch):
        # sample elevation angles
        batch = torch.utils.data.default_collate(batch)
        B = batch["index"].shape[0]
        batch.update(
            {
                "batch_size": B,
                "elevation": torch.zeros(B, device=self.device),
                "azimuth": torch.zeros(B, device=self.device),
                "camera_distances": torch.ones(B, device=self.device),
                "c2w": torch.eye(4, device=self.device)[None].repeat(B, 1, 1),
                "mvp_mtx": None,
            }
        )

        return batch


@register("placeholder-datamodule")
class PlaceholderDataModule(pl.LightningDataModule):
    cfg: PlaceholderDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(PlaceholderDataModuleConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = PlaceholderIterableDataset(self.cfg)
        if stage in [None, "fit", "validate"]:
            self.val_dataset = PlaceholderDataset(self.cfg, "val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = PlaceholderDataset(self.cfg, "test")

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size, collate_fn=None) -> DataLoader:
        return DataLoader(
            dataset,
            # very important to disable multi-processing if you want to change self attributes at runtime!
            # (for example setting self.width and self.height in update_step)
            num_workers=0,  # type: ignore
            batch_size=batch_size,
            collate_fn=collate_fn,
        )

    def train_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.train_dataset, batch_size=None, collate_fn=self.train_dataset.collate
        )

    def val_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.val_dataset, batch_size=1, collate_fn=self.val_dataset.collate
        )
        # return self.general_loader(self.train_dataset, batch_size=None, collate_fn=self.train_dataset.collate)

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=1, collate_fn=self.test_dataset.collate
        )

    def predict_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=1, collate_fn=self.test_dataset.collate
        )
