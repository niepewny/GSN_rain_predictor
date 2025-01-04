import h5py
import torch
from torch.utils.data import DataLoader, Dataset, Subset
import pytorch_lightning as pl
from torch.nn import functional as F
import os
from torchvision import transforms
from data_exploration import visualize_batch_tensor_interactive
import os
os.environ["HDF5_USE_FILE_LOCKING"]='FALSE'




all_file_paths_2019 = [
    "../../data/2019/SEVIR_IR069_RANDOMEVENTS_2019_0101_0430.h5", #val
    "../../data/2019/SEVIR_IR069_RANDOMEVENTS_2019_0501_0831.h5", #tesut
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


class SEVIR_dataset(Dataset):
    """
    Dataset ładujący dane SEVIR z wielu plików HDF5.
    """
    def __init__(self, file_paths, step, width, height):
        super().__init__()
        # Convert relative paths to absolute paths
        self.file_paths = [os.path.abspath(path) for path in file_paths]

        self.samples_per_file = []
        self._cumulative_indices = []
        self.step = step
        self.width = width
        self.height = height

        current_cum = 0
        for path in self.file_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")

            with h5py.File(path, 'r') as f:
                data_shape = f['ir069'].shape
                x_size = data_shape[0]
                self.samples_per_file.append(x_size)
                current_cum += x_size
                self._cumulative_indices.append(current_cum)

    def __len__(self):
        return self._cumulative_indices[-1]

    def __getitem__(self, index):
        # Znajdujemy indeks pliku, w którym znajduje się żądana próbka
        file_idx = self._find_file_index(index)

        # Obliczamy lokalny indeks w znalezionym pliku:
        # - Dla pierwszego pliku (idx=0) indeks lokalny = indeks globalny
        # - Dla kolejnych plików odejmujemy sumę próbek z poprzednich plików
        if file_idx == 0:
            local_index = index
        else:
            local_index = index - self._cumulative_indices[file_idx - 1]

        file_path = self.file_paths[file_idx]

        '''
        pipeline przetwarzania próbek:
        1. lazy loading na podstawie indeksu lokalnego
        2. zamiana z 192x192x49 na 49x192x192
        3. pobranie co ntej klatki czasowej - np przy kroku 2 zamiana na 25x192x192
        4. zmiana rozmiaru na np. 25 x height x width
        5. normalizacja z zakresu 0-255 na 0-1
        '''
        try:
            # otwarcie sampla z pliku za pomocą indeksu lokalnego(własciwego dla danego pliku)
            with h5py.File(file_path, 'r') as f:
                sample = f['ir069'][local_index]
                sample = torch.tensor(sample, dtype=torch.float32)
                # zamienia z 192x192x49 na 49x192x192
                permuted_sample = sample.permute(2, 0, 1)
                # przy kroku 2 zamienia na 25x192x192
                permuted_sample_with_step = self._get_sample_with_step(permuted_sample, self.step)
                # zmiana rozmiaru na np. 25 x height x width
                if self.width != 192 or self.height != 192:
                    resize = transforms.Resize((self.height, self.width), antialias=True)
                    permuted_sample_with_step_resized = resize(permuted_sample_with_step)
                else:
                    permuted_sample_with_step_resized = permuted_sample_with_step
                # normalizacja z zakresu 0-255 na 0-1
                permuted_sample_with_step_resized_normalized = permuted_sample_with_step_resized / 1000
                return permuted_sample_with_step_resized_normalized
        except Exception as e:
            print(f"Error loading file {file_path} at index {local_index}")
            raise e

    def _find_file_index(self, index):
        for i, cum in enumerate(self._cumulative_indices):
            if index < cum:
                return i
        raise IndexError(f"Index {index} out of range {self.__len__()}")

    def _get_sample_with_step(self,tensor,step):
        """
        Zwraca klatki z tensora z krokiem step.
        """
        frames = []
        for i in range(0, tensor.shape[0], step):
            frame = tensor[i]
            frames.append(frame)
        return torch.stack(frames)

class ConvLSTMSevirDataModule(pl.LightningDataModule):
    """
    DataModule dla projektu z convLSTM na zbiorze SEVIR (kanał IR069).
    W metodzie setup 3 dataset-y (train, val, test)
    Domyślnie przypisujemy podział plików test,train,val jak na górze pliku.
    """

    def __init__(
        self,
        # parametry transformacji danych
        step,
        width,
        height,
        # przypisanie plików do zbiorów
        train_files=train_files,
        val_files=validate_files,
        test_files=test_files,
        file_paths=all_file_paths,
        # parametry DataLoadera
        batch_size=4,
        num_workers=2
    ):
        super().__init__()
        self.train_files = train_files
        self.val_files = val_files
        self.test_files = test_files
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.file_paths = file_paths
        self.step = step
        self.width = width
        self.height = height

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None


    def prepare_data(self):
        # zakładamy że pliki są już lokalnie.
        # pobrane komendą z README
        pass

    def setup(self, stage=None):
        # Tworzymy dataset-y.
        if stage == 'fit' or stage is None:
            self.train_dataset = SEVIR_dataset(self.train_files, self.step, self.width, self.height)
            self.val_dataset   = SEVIR_dataset(self.val_files, self.step, self.width, self.height)

        if stage == 'test' or stage is None:
            self.test_dataset  = SEVIR_dataset(self.test_files, self.step, self.width, self.height)

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


if __name__ == "__main__":
    # przykład użycia oba przykłady są równoważne
    # tylko jeden jest na pytorch a drugi na pytorch lighnting

    ''' pytorch dataset '''
    # przyjmuje step oraz szerokość i wysokość obrazka
    dataset = SEVIR_dataset(train_files, 3, 128, 128)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
    fist_sample = next(iter(dataloader))
    print("data loader",fist_sample.shape) # zwraca torch.Size([10, 17, 128, 128])
    # visualize_batch_tensor_interactive(fist_sample, 0, "SEVIR dataset")

    ''' pytorch lightning datamodule '''
    # przykład użycia
    dm = ConvLSTMSevirDataModule(step=3, width=128, height=128, batch_size=4, num_workers=2)
    dm.setup('fit')
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    print("data module",batch.shape) # zwraca torch.Size([4, 17, 128, 128
    visualize_batch_tensor_interactive(batch, 0, "SEVIR dataset")
