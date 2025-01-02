import pandas as pd
import h5py
import torch
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider
from matplotlib.widgets import Slider
import os
from torch.utils.data import DataLoader, Dataset, Subset
import pytorch_lightning as pl


class ConvLSTMSevirDataModule(pl.LightningDataModule):
    """
    DataModule dla projektu z convLSTM na zbiorze SEVIR (kanał IR069).
    W metodzie setup 3 dataset-y (train, val, test)
    bazując na zewn. listach plików. Dodatkowo definiujemy metody:
      - get_*_data_skip(skip=...)  => generuje Subset z co skip-tym indeksem
      - get_*_data_range(start_idx, count, step=1) => Subset od start_idx
        (count próbek, co step).
    W obszarze test/val/train mamy 49 kroków czasowych (dim=49).
    """

    def __init__(
        self,
        train_files,
        val_files,
        test_files,
        batch_size=8,
        num_workers=2
    ):
        super().__init__()
        self.train_files = train_files
        self.val_files = val_files
        self.test_files = test_files
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        # zakładamy że pliki są już lokalnie.
        pass

    def setup(self, stage=None):
        # Tworzymy dataset-y.
        if stage == 'fit' or stage is None:
            self.train_dataset = SevirDataset(self.train_files)
            self.val_dataset   = SevirDataset(self.val_files)

        if stage == 'test' or stage is None:
            self.test_dataset  = SevirDataset(self.test_files)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )


    # Metody do wyciągania sub-datasetów co skip lub w określonym zakresie
    # Zwraca Subset z sampli co skip w zbiorze treningowym.
    def get_train_data_skip(self, skip=1):

        indices = range(0, len(self.train_dataset), skip)
        return Subset(self.train_dataset, indices)

    def get_val_data_skip(self, skip=1):
        indices = range(0, len(self.val_dataset), skip)
        return Subset(self.val_dataset, indices)

    def get_test_data_skip(self, skip=1):
        indices = range(0, len(self.test_dataset), skip)
        return Subset(self.test_dataset, indices)

    def get_train_data_range(self, start_idx=0, count=10, step=1):
        end_idx = start_idx + count * step
        indices = range(start_idx, end_idx, step)
        return Subset(self.train_dataset, indices)

    def get_val_data_range(self, start_idx=0, count=10, step=1):
        end_idx = start_idx + count * step
        indices = range(start_idx, end_idx, step)
        return Subset(self.val_dataset, indices)

    def get_test_data_range(self, start_idx=0, count=10, step=1):
        end_idx = start_idx + count * step
        indices = range(start_idx, end_idx, step)
        return Subset(self.test_dataset, indices)



if __name__ == "__main__":
    all_file_paths_2019 = [
        "../../data/2019/SEVIR_IR069_RANDOMEVENTS_2019_0101_0430.h5", #val
        "../../data/2019/SEVIR_IR069_RANDOMEVENTS_2019_0501_0831.h5", #test
        "../../data/2019/SEVIR_IR069_RANDOMEVENTS_2019_0901_1231.h5", #train
        "../../data/2019/SEVIR_IR069_STORMEVENTS_2019_0101_0630.h5", # val
        "../../data/2019/SEVIR_IR069_STORMEVENTS_2019_0701_1231.h5"  #test
    ]
    all_file_paths_2018 = [ # train
        "../../data/2018/SEVIR_IR069_RANDOMEVENTS_2018_0101_0430.h5",
        "../../data/2018/SEVIR_IR069_RANDOMEVENTS_2018_0501_0831.h5",
        "../../data/2018/SEVIR_IR069_RANDOMEVENTS_2018_0901_1231.h5",
        "../../data/2018/SEVIR_IR069_STORMEVENTS_2018_0701_1231.h5",
        "../../data/2018/SEVIR_IR069_STORMEVENTS_2018_0101_0630.h5"
    ]
    all_file_paths = all_file_paths_2018 + all_file_paths_2019

    #test val train split(each includes at least one storm file)
    train_files = [all_file_paths_2018,all_file_paths_2019[2]]

    validate_files = [all_file_paths_2019[0],all_file_paths_2019[3]]

    test_files = [ all_file_paths_2019[1],all_file_paths_2019[4]]

    dm = ConvLSTMSevirDataModule(
        train_files=train_files,
        val_files=validate_files,
        test_files=test_files,
        batch_size=2,
        num_workers=4
    )

    dm.setup('fit')
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
