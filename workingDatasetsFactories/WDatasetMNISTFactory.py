from torchvision import transforms

from workingDatasets.MNISTWorkingDataset import MNISTWorkingDataset
from workingDatasets.WorkingDataset import WorkingDataset
from workingDatasetsFactories.WorkingDatasetAbstractFactory import WorkingDatasetAbstractFactory
from xmlComponents.WorkingDatasetXML import WorkingDatasetXML


class WDatasetMNISTFactory(WorkingDatasetAbstractFactory):

    def _instantiateDataset(self, workingDatasetXML: WorkingDatasetXML, compose: transforms.Compose, composeWithoutNormalization: transforms.Compose) -> WorkingDataset:
        workingDataset = MNISTWorkingDataset(workingDatasetXML.getStoragePath(), workingDatasetXML.getLimitRawSize(), compose, composeWithoutNormalization)
        return workingDataset


