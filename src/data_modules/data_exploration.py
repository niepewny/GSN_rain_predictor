import pandas as pd
import h5py
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider
from matplotlib.widgets import Slider

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
    sample0 = sevirDataSet.__getitem__(random)
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

if __name__ == "__main__":

    print_all_files_info(all_file_paths)
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
    #     sample0 = sevirDataSet.__getitem__(0)
    #     print("data shape:",sample0.shape, "\n")
    #     first_frame = sample0[:, :, 0]

    #     # visualize_frame(first_frame)
    #     visualize_tensor_interactive(sample0,f"pierwszy z {file}")
    #     # visualize_tensor_interactive(sample1,f"drugi z {file}")
    #     # visualize_tensor_interactive(sample2,f"trzeci z {file}")
