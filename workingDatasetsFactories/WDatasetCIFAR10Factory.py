from torchvision import transforms

from workingDatasets.CIFAR10WorkingDataset import CIFAR10WorkingDataset
from workingDatasets.WorkingDataset import WorkingDataset
from workingDatasetsFactories.WorkingDatasetAbstractFactory import WorkingDatasetAbstractFactory
from xmlComponents.WorkingDatasetXML import WorkingDatasetXML

class WDatasetCIFAR10Factory(WorkingDatasetAbstractFactory):

    def _instantiateDataset(self, workingDatasetXML: WorkingDatasetXML, compose: transforms.Compose, composeWithoutNormalization: transforms.Compose) -> WorkingDataset:
        workingDataset = CIFAR10WorkingDataset(workingDatasetXML.getStoragePath(), workingDatasetXML.getLimitRawSize(), compose, composeWithoutNormalization)
        return workingDataset



