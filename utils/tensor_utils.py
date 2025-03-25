import os.path
import shutil

import numpy as np
import torch
from matplotlib.figure import Figure
from PIL import Image

#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

def values_target(size: tuple, value: float) -> torch.Tensor:
    result = torch.full(size=size, fill_value=value).to(device=get_device())
    return result

def get_device():
    global device
    return device

def normalize_tensor(vector: torch.tensor) -> torch.tensor:
    """ normalize np array to the range of [0,1] and returns as float32 values """
    vector -= vector.min()
    vector /= vector.max()
    vector[torch.isnan(vector)] = 0
    return vector.type(torch.float32)

def figure_to_tensor_own_impl(fig: Figure) -> torch.Tensor:
    fig.canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    image_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float() / 255.0
    return image_tensor

def normalise_tensor_uint8(vector: torch.Tensor) -> torch.Tensor:
    # normalise [0, 1] -> [0, 255] torch.unit8
    vector = vector * 255
    vector = vector.to(torch.uint8)
    return vector

'''
counterfeitImages: shape [N, C, H, W] normalised in [-1, 1]
'''
def save_tanhElems_as_Image(counterfeitImages: torch.Tensor, savingPath: str) -> None:
    N = counterfeitImages.shape[0]
    for i in range(N):
        sample = counterfeitImages[i]
        sample = np.transpose(sample, (1, 2, 0))
        sample = ((sample / 2) + 0.5) * 255
        sample = sample.astype(np.uint8)
        image = Image.fromarray(sample)
        image.save(f'{savingPath}/{i}.jpg')

'''
realImages: shape [N, C, H, W] normalised in [0, 1]
'''
def save_01Elems_as_Image(realImages: torch.Tensor, savingPath: str) -> None:
    N = realImages.shape[0]
    for i in range(N):
        sample = realImages[i].detach().cpu().numpy()
        sample = np.transpose(sample, (1, 2, 0))
        sample = sample * 255
        sample = sample.astype(np.uint8)
        image = Image.fromarray(sample)
        image.save(f'{savingPath}/{i}.jpg')

def resetDirectory(pathDir: str) -> None:
    if os.path.exists(pathDir) and os.path.isdir(pathDir):
        try:
            shutil.rmtree(pathDir)
            print(f"The directory '{pathDir}' has been deleted successfully.")
        except Exception as e:
            print(f"Failed to delete the directory: {e}")
    os.makedirs(pathDir, exist_ok=True)



