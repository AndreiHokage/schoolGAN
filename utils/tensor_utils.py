import numpy as np
import torch
from matplotlib.figure import Figure

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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

