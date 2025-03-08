import os.path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.datasets as datasets
from torch.utils.data.dataset import _T_co

from utils.tensor_utils import get_device
from workingDatasets.WorkingDataset import WorkingDataset


class MNISTWorkingDataset(WorkingDataset):

    def __init__(self, storage_path: str, limitRawSize: int, compose: transforms.Compose, composeWithoutNormalization: transforms.Compose):
        super().__init__(limitRawSize)
        self.__mnistDataset = datasets.MNIST(root=storage_path, transform=compose, train=True, download=True)
        self.__mnistDataset = self._shrinkDataset(self.__mnistDataset)
        # no normalisation, the images are in [0, 1]
        self.__evalMnistDataset: Dataset = datasets.MNIST(root=os.path.join(storage_path, "eval"), transform=composeWithoutNormalization, train=False, download=True)

    def __len__(self):
        return len(self.__mnistDataset)

    def __getitem__(self, index) -> _T_co:
        return self.__mnistDataset[index]

    def generateRealSamples(self, numSamples=1) -> torch.Tensor:
        shrinkEvalDataset = self._shrinkDataset(self.__evalMnistDataset)
        loader = DataLoader(shrinkEvalDataset,  batch_size=numSamples, shuffle=False)
        realBatch = next(iter(loader))[0].to(device=get_device())
        return realBatch




