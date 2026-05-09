from typing import Dict, List, Optional, NoReturn
import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from data.audiotext_dataset import AudioTextDataset


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dataset: object,
        val_dataset: object = None,
        batch_size: int = 12,
        num_workers: int = 4
    ):
        r"""Data module. To get one batch of data:

        code-block:: python

            data_module.setup()

            for batch_data_dict in data_module.train_dataloader():
                print(batch_data_dict.keys())
                break

        Args:
            train_sampler: Sampler object
            train_dataset: Dataset object
            num_workers: int
            distributed: bool
        """
        super().__init__()
        self._train_dataset = train_dataset
        self._val_dataset = val_dataset
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.collate_fn = collate_fn


    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        pass

    def setup(self, stage: Optional[str] = None) -> NoReturn:
        r"""called on every device."""

        # make assignments here (val/train/test split)
        # called on every process in DDP

        self.train_dataset = self._train_dataset
        self.val_dataset = self._val_dataset
        
        
    def train_dataloader(self) -> torch.utils.data.DataLoader:
        r"""Get train loader."""
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=False,
            shuffle=True
        )

        return train_loader

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        r"""Get val loader."""
        if self.val_dataset is None:
            return None
            
        val_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=False,
            shuffle=False
        )

        return val_loader

    def test_dataloader(self):
        # test_split = Dataset(...)
        # return DataLoader(test_split)
        pass

    def teardown(self, stage: Optional[str] = None):
        # clean up after fit or test
        # called on every process in DDP
        pass


def collate_fn(list_data_dict):
    r"""Collate mini-batch data to inputs and targets for training.

    Args:
        list_data_dict: e.g., [
            {
                'text': 'a sound of dog',
                'waveform': (1, samples),
                'modality': 'audio_text'
            }
            ...
            ]
    Returns:
        data_dict: e.g. 
            'audio_text': {
                'text': ['a sound of dog', ...]
                'waveform': (batch_size, 1, samples)
        }
    """
    
    at_list_data_dict = [data_dict for data_dict in list_data_dict if data_dict.get('modality') == 'audio_text']

    at_data_dict = {}
    
    if len(at_list_data_dict) > 0:
        for key in at_list_data_dict[0].keys():
            at_data_dict[key] = [at_data_dict[key] for at_data_dict in at_list_data_dict]
            if key == 'waveform' or key == 'mixture':
                at_data_dict[key] = torch.stack(at_data_dict[key])
            elif key == 'text':
                at_data_dict[key] = [text for text in at_data_dict[key]]

    
    data_dict = {
        'audio_text': at_data_dict
    }
    
    return data_dict