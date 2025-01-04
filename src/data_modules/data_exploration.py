import pandas as pd
import h5py
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider
from matplotlib.widgets import Slider
# from SEVIR_data_loader import SEVIR_dataset
from torch.utils.data import DataLoader, Dataset, Subset

#todo:
    # parametryzacja resize
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

class SEVIRDataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        with h5py.File(self.file_path, 'r') as f:
            self.data_length = f['ir069'].shape[0]
            self.ids = f['id'][:]

    def __len__(self):
        return self.data_length

    def __getitem__(self, idx):
        with h5py.File(self.file_path, 'r') as f:
            sample = f['ir069'][idx]
            sample = torch.tensor(sample, dtype=torch.float32)
        return sample

def visualize_frame(tensor):
    # Convert tensor to numpy array from PyTorch tensor
    if torch.is_tensor(tensor):
        tensor = tensor.numpy()

    # Plot the tensor using matplotlib with blue to green to red color map
    plt.imshow(tensor, cmap='seismic')
    plt.colorbar()
    plt.title("2D Tensor Visualization")
    plt.show()

def visualize_batch_tensor_interactive(tensor, batch_idx=0, name=None):
    """
    Creates an interactive visualization of a 4D tensor with shape [batch, frames, height, width]
    Parameters:
        tensor: PyTorch tensor of shape [batch, frames, height, width]
        batch_idx: Which sample from the batch to visualize
        name: Optional name for the visualization
    """
    # Convert tensor to numpy if it's a PyTorch tensor
    if torch.is_tensor(tensor):
        # Select specific sample from batch and convert to numpy
        tensor = tensor[batch_idx].numpy()  # Now shape is [frames, height, width]

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(bottom=0.2)  # Make space for the slider

    # Display initial frame
    frame_idx = 0
    # For tensor [frames, height, width], we display frame_idx slice
    img = ax.imshow(tensor[frame_idx], cmap='viridis')
    plt.colorbar(img)

    # Create slider axis and slider
    ax_slider = plt.axes([0.1, 0.05, 0.65, 0.03])  # [left, bottom, width, height]
    slider = Slider(
        ax=ax_slider,
        label='Frame',
        valmin=0,
        valmax=tensor.shape[0] - 1,  # Number of frames is first dimension now
        valinit=frame_idx,
        valstep=1
    )

    # Set title
    if name:
        title = ax.set_title(f'{name} - Frame {frame_idx} / {tensor.shape[0]-1} (Batch {batch_idx})')
    else:
        title = ax.set_title(f'Frame {frame_idx} / {tensor.shape[0]-1} (Batch {batch_idx})')

    def update(val):
        frame = int(slider.val)
        img.set_array(tensor[frame])
        if name:
            title.set_text(f'{name} - Frame {frame} / {tensor.shape[0]-1} (Batch {batch_idx})')
        else:
            title.set_text(f'Frame {frame} / {tensor.shape[0]-1} (Batch {batch_idx})')
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


def visualize_tensor_interactive(tensor,name):
    """
    Creates an interactive visualization of a 3D tensor with shape [height, width, frames]
    Parameters:
        tensor: PyTorch tensor or numpy array of shape [height, width, frames]
    """
    # Convert tensor to numpy if it's a PyTorch tensor
    if torch.is_tensor(tensor):
        tensor = tensor.numpy()

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(bottom=0.2)  # Make space for the slider

    # Display initial frame
    frame_idx = 0
    img = ax.imshow(tensor[:, :, frame_idx], cmap='viridis')
    plt.colorbar(img)

    # Create slider axis and slider
    ax_slider = plt.axes([0.1, 0.05, 0.65, 0.03])  # [left, bottom, width, height]
    slider = Slider(
        ax=ax_slider,
        label='Frame',
        valmin=0,
        valmax=tensor.shape[2] - 1,
        valinit=frame_idx,
        valstep=1
    )

    # Title with frame information
    title = ax.set_title(f'Frame {frame_idx} / {tensor.shape[2]-1}')

    # Update function for the slider
    if name:
        title = ax.set_title(f'{name} - Frame {frame_idx} / {tensor.shape[2]-1}')
    else:
        title = ax.set_title(f'Frame {frame_idx} / {tensor.shape[2]-1}')

        # Modified update function to maintain the name in the title
    def update(val):
        frame = int(slider.val)
        img.set_array(tensor[:, :, frame])
        if name:
            title.set_text(f'{name} - Frame {frame} / {tensor.shape[2]-1}')
        else:
            title.set_text(f'Frame {frame} / {tensor.shape[2]-1}')
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()

def visualize_random_sample(file_name):
    sevirDataSet = SEVIRDataset(file_name)

    random = torch.randint(0, 552, (1,)).item()
    sample0 = SEVIRDataset.__getitem__(random)
    visualize_tensor_interactive(sample0,f"Random sample(id:{random}) from: {file_name}")


def print_data_info(file_path):
    with h5py.File(file_path, 'r') as f:
        # print(f.keys())
        # print(f['ir069'].shape)
        print(f['id'].shape,"\n")
        return f['id'].shape
        # print(f['id'][:])

def print_all_files_info(all_file_paths):
    samples_sum = 0
    for file in all_file_paths:
        print(f"FILE {file}")
        num_samples = print_data_info(file)
        samples_sum += num_samples[0]
    print(f"Total files: {len(all_file_paths)}")
    print(f"Total samples: {samples_sum}")
    print(f"size (X, 192, 192, 49)")

def compare_storm_to_randomevents():
    file_path_randomevents = "../../data/2018/SEVIR_IR069_RANDOMEVENTS_2018_0101_0430.h5"
    file_path_storm = "../../data/2018/SEVIR_IR069_STORMEVENTS_2018_0101_0630.h5"

    visualize_random_sample(file_path_storm)
    visualize_random_sample(file_path_randomevents)

def analyze_data_distribution(dataset, num_batches=100, batch_size=32):
    """
    Analizuje rozkład wartości w datasecie.

    Args:
        dataset: Dataset do przeanalizowania
        num_batches: Ile batchy przeanalizować
        batch_size: Wielkość batcha
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # Inicjalizacja list na wartości
    all_mins = []
    all_maxs = []
    all_values = []

    print("Zbieranie statystyk...")

    # Zbieranie wartości z próbek
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break

        batch_min = batch.min().item()
        batch_max = batch.max().item()

        all_mins.append(batch_min)
        all_maxs.append(batch_max)

        # Zbieranie wszystkich wartości dla histogramu
        all_values.extend(batch.flatten().tolist())

        if i % 10 == 0:
            print(f"Przetworzono {i}/{num_batches} batchy")

    # Obliczanie globalnych statystyk
    global_min = min(all_mins)
    global_max = max(all_maxs)

    # Tworzenie histogramu
    plt.figure(figsize=(12, 6))
    plt.hist(all_values, bins=10, edgecolor='black')
    plt.title('Rozkład wartości w datasecie')
    plt.xlabel('Wartość')
    plt.ylabel('Liczba wystąpień')
    plt.grid(True, alpha=0.3)

    # Dodanie statystyk do wykresu
    stats_text = f'Min: {global_min:.2f}\nMax: {global_max:.2f}'
    plt.text(0.95, 0.95, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Obliczanie przedziałów
    bins = np.linspace(global_min, global_max, 11)
    print("\nPrzedziały wartości:")
    for i in range(len(bins)-1):
        print(f"Przedział {i+1}: [{bins[i]:.2f}, {bins[i+1]:.2f})")

    print(f"\nWartość minimalna: {global_min:.2f}")
    print(f"Wartość maksymalna: {global_max:.2f}")

    plt.show()

if __name__ == "__main__":


    analyze_data_distribution(SEVIRDataset(all_file_paths[0]), num_batches=100, batch_size=32)

    # print_all_files_info(all_file_paths)
    # compare_storm_to_randomevents()

    # dane w formacie: shape=(553, 192, 192, 49)
    # 533 próbki, 192x192 pikseli, 49 klatek czasowych co 5 min
    # for file in all_file_paths_2018:
    #     print(file)
    #     sevirDataSet = SEVIRDataset(file)

    #     # przypadkoowy wybór próbki
    #     random = torch.randint(0, 49, (1,)).item()
    #     sample0 = sevirDataSet.__getitem__(random)


    #     print("data length:",evirDataSet.data_length,"\n")
    #     sample0 = SEVIRDataset.__getitem__(0)
    #     print("data shape:",sample0.shape, "\n")
    #     first_frame = sample0[:, :, 0]

    #     # visualize_frame(first_frame)
    #     visualize_tensor_interactive(sample0,f"pierwszy z {file}")
    #     # visualize_tensor_interactive(sample1,f"drugi z {file}")
    #     # visualize_tensor_interactive(sample2,f"trzeci z {file}")
