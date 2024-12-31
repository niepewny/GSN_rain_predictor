import pandas as pd
import h5py
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider
from matplotlib.widgets import Slider


class SEVIRDataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        with h5py.File(self.file_path, 'r') as f:
            self.data_length = f['ir107'].shape[0]
            self.ids = f['id'][:]

    def __len__(self):
        return self.data_length

    def __getitem__(self, idx):
        with h5py.File(self.file_path, 'r') as f:
            sample = f['ir107'][idx]
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

def visualize_tensor_interactive(tensor):
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
    def update(val):
        frame = int(slider.val)
        img.set_array(tensor[:, :, frame])
        title.set_text(f'Frame {frame} / {tensor.shape[2]-1}')
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


if __name__ == "__main__":
    file_path = "../../data/2018/SEVIR_IR107_RANDOMEVENTS_2018_0101_0430.h5"
    # dane w formacie: shape=(553, 192, 192, 49)
    # 533 próbki, 192x192 pikseli, 49 klatek czasowych co 5 min
    sevirDataSet = SEVIRDataset(file_path)
    print("data length:",sevirDataSet.data_length,"\n")
    sample0 = sevirDataSet.__getitem__(0)
    sample1 = sevirDataSet.__getitem__(1)
    sample2 = sevirDataSet.__getitem__(2)

    print("data shape:",sample0.shape, "\n")
    first_frame = sample0[:, :, 0]

    # visualize_frame(first_frame)
    visualize_tensor_interactive(sample0)
    visualize_tensor_interactive(sample1)
    visualize_tensor_interactive(sample2)
