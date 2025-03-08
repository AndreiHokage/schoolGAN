import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.datasets as datasets
from torch.utils.data.dataset import _T_co, Dataset

from utils.tensor_utils import get_device
from workingDatasets.WorkingDataset import WorkingDataset

class CIFAR10WorkingDataset(WorkingDataset):

    def __init__(self, storage_path: str, limitRawSize: int, compose: transforms.Compose, composeWithoutNormalization: transforms.Compose):
        super().__init__(limitRawSize)
        self.__cifar10Dataset = datasets.CIFAR10(root=storage_path, transform=compose, train=True, download=True)
        self.__cifar10Dataset = self._shrinkDataset(self.__cifar10Dataset)
        self.__evalCifar10Dataset: Dataset = datasets.CIFAR10(root=os.path.join(storage_path, "eval"), transform=composeWithoutNormalization, train=False, download=True)

    def __len__(self):
        return len(self.__cifar10Dataset)

    def __getitem__(self, index) -> _T_co:
        return self.__cifar10Dataset[index]

    def generateRealSamples(self, numSamples=1) -> torch.Tensor:
        shrinkEvalDataset = self._shrinkDataset(self.__evalCifar10Dataset)
        loader = DataLoader(shrinkEvalDataset,  batch_size=numSamples, shuffle=False)
        realBatch = next(iter(loader))[0].to(device=get_device())
        return realBatch
