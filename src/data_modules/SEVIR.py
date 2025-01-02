import h5py
import torch
from torch.utils.data import DataLoader, Dataset, Subset
import pytorch_lightning as pl
from data_exploration import visualize_tensor_interactive

TRAIN_ID = 0
TEST_ID = 0
VAL_ID = 0

class SevirDataset(Dataset):
    """
    Dataset ładujący dane SEVIR z wielu plików HDF5.
    Przyjmuje listę ścieżek do plików .h5, w każdym pliku
    interesuje nas dataset `ir069` (np. shape: (X, 192, 192, 49)).

    Zakładamy:
      - Każdy plik ma wymiary (num_samples_i, 192, 192, 49)
      - Z puli plików sumarycznie tworzymy logiczny "jeden dataset",
        gdzie index globalny -> (konkretny plik, index w pliku).
      - Otwieramy plik w locie przy __getitem__ (lazy loading).
    """
    def __init__(self, file_paths):
        super().__init__()
        self.file_paths = file_paths

        # Liczymy ile jest łącznie próbek w każdym pliku, by móc
        # mapować globalny index -> (który plik, wewn. index).
        self.samples_per_file = []
        self._cumulative_indices = []



        current_cum = 0
        for path in self.file_paths:
            with h5py.File(path, 'r') as f:
                # tu 'ir069' jest przykładową nazwą datasetu w pliku .h5
                data_shape = f['ir069'].shape  # np. (X, 192,192,49)
                x_size = data_shape[0]         # X
            self.samples_per_file.append(x_size)
            current_cum += x_size
            self._cumulative_indices.append(current_cum)

    def __len__(self):
        return self._cumulative_indices[-1]

    def __getitem__(self, index):
        # Znajdujemy, w którym pliku jest dany index
        file_idx = self._find_file_index(index)
        # index w danym pliku (lokalny)
        if file_idx == 0:
            local_index = index
        else:
            local_index = index - self._cumulative_indices[file_idx - 1]

        file_path = self.file_paths[file_idx]
        with h5py.File(file_path, 'r') as f:
            sample = f['ir069'][local_index]  # shape: (192,192,49)
            sample = torch.tensor(sample, dtype=torch.float32)
            # Ewentualnie można permutować wymiary tak, by mieć (49,1,192,192)
            # np. sample = sample.permute(2, 0, 1).unsqueeze(1)
            # w zależności od potrzeb ConvLSTM
        return sample

    def _find_file_index(self, index):
        """
        Z binary search lub prostą pętlą (dla uproszczenia pętla),
        zwracamy numer pliku, w którym leży sample o globalnym indexie.
        """
        for i, cum in enumerate(self._cumulative_indices):
            if index < cum:
                return i
        raise IndexError(f"Index {index} out of range {self.__len__()}")


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

        # Pobieranie kolejnego przykładu do trenowania zwiększa id więc
        # pobiera się kolejne przykłady
        self.TRAIN_ID = 0
        self.TEST_ID = 0
        self.VAL_ID = 0

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
    def get_train_sample(self):
        '''
        Returns full train sample 49x192x192
        '''
        indices = range(0, len(self.train_dataset))
        subset = Subset(self.train_dataset, indices)
        sample = subset[self.TRAIN_ID]
        self.TRAIN_ID += 1
        return sample

    def get_train_data_skip(self, num_frames, skip=1):

        if num_frames*skip > 49:
            raise ValueError("Skip too large(over 49 frames)")
        indices = range(0, len(self.train_dataset), skip)
        subset = Subset(self.train_dataset, indices)
        sample = subset[self.TRAIN_ID]
        self.TRAIN_ID += 1
        sampled_frames_tensor = sample[:,:,0:49:skip]
        return sampled_frames_tensor

    def get_val_sample(self):
        '''
        Returns full validation sample 49x192x192
        '''
        indices = range(0, len(self.val_dataset))
        subset = Subset(self.val_dataset, indices)
        sample = subset[self.VAL_ID]
        self.VAL_ID += 1
        return sample

    def get_val_data_skip(self, num_frames, skip=1):
        '''
        Returns sampled validation data with reduced frames [num_frames x 192 x 192]
        '''
        if num_frames * skip > 49:
            raise ValueError("Skip too large (over 49 frames)")
        indices = range(0, len(self.val_dataset), skip)
        subset = Subset(self.val_dataset, indices)
        sample = subset[self.VAL_ID]
        self.VAL_ID += 1
        sampled_frames_tensor = sample[:, :, 0:49:skip]
        return sampled_frames_tensor

    def get_test_sample(self):
        '''
        Returns full test sample 49x192x192
        '''
        indices = range(0, len(self.test_dataset))
        subset = Subset(self.test_dataset, indices)
        sample = subset[self.TEST_ID]
        self.TEST_ID += 1
        return sample

    def get_test_data_skip(self, num_frames, skip=1):
        '''
        Returns sampled test data with reduced frames [num_frames x 192 x 192]
        '''
        if num_frames * skip > 49:
            raise ValueError("Skip too large (over 49 frames)")
        indices = range(0, len(self.test_dataset), skip)
        subset = Subset(self.test_dataset, indices)
        sample = subset[self.TEST_ID]
        self.TEST_ID += 1
        sampled_frames_tensor = sample[:, :, 0:49:skip]
        return sampled_frames_tensor




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
    train_files = all_file_paths_2018 + [all_file_paths_2019[2]]

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

    # test train
    # dm.TRAIN_ID = 0
    # sample = dm.get_train_sample()
    # visualize_tensor_interactive(sample, "Subset 0")
    #
    # #cofnięcie countera żeby dwa razy zobrazować to samo
    # dm.TRAIN_ID=0
    # sampled_tensor_skip = dm.get_train_data_skip(20, 2)
    # print(sampled_tensor_skip)
    # visualize_tensor_interactive(sampled_tensor_skip, "Subset with skip")

    dm.setup('fit')
    dm.VAL_ID = 0
    sample = dm.get_val_sample()
    visualize_tensor_interactive(sample, "Subset 0")

    # cofnięcie countera żeby dwa razy zobrazować to samo
    dm.VAL_ID = 0
    sampled_tensor_skip = dm.get_val_data_skip(20, 2)
    print(sampled_tensor_skip)
    visualize_tensor_interactive(sampled_tensor_skip, "Subset with skip")

    # test test
    # dm.setup('test')
    # dm.TEST_ID = 0
    # sample = dm.get_test_sample()
    # visualize_tensor_interactive(sample, "Subset 0")
    #
    # # cofnięcie countera żeby dwa razy zobrazować to samo
    # dm.TEST_ID = 0
    # sampled_tensor_skip = dm.get_test_data_skip(20, 2)
    # print(sampled_tensor_skip)
    # visualize_tensor_interactive(sampled_tensor_skip, "Subset with skip")