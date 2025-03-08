import torch
from torch.utils.data import Dataset, Subset


class WorkingDataset(Dataset):

    def __init__(self, limitRawSize: int):
        self.__limitRawSize = limitRawSize

    def _shrinkDataset(self, dataset: Dataset) -> Dataset:
        if self.__limitRawSize != - 1:
            return Subset(dataset, list(range(self.__limitRawSize)))
        else:
            return dataset

    def generateRealSamples(self, numSamples=1) -> torch.Tensor:
        pass

