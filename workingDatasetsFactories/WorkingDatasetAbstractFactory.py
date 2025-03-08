import ast

from workingDatasets.WorkingDataset import WorkingDataset
from xmlComponents.WorkingDatasetXML import WorkingDatasetXML
from torchvision import transforms


class WorkingDatasetAbstractFactory:

    def createDataset(self, workingDatasetXML: WorkingDatasetXML) -> WorkingDataset:
        compose = self._instantiateTransformer(workingDatasetXML)
        composeWithoutNormalization = self._instantiateTransformerWithoutNormalization(workingDatasetXML)
        workingDataset = self._instantiateDataset(workingDatasetXML, compose, composeWithoutNormalization)
        return workingDataset

    def _instantiateDataset(self, workingDatasetXML: WorkingDatasetXML, compose: transforms.Compose, composeWithoutNormalization: transforms.Compose) -> WorkingDataset:
        pass

    def _instantiateTransformer(self, workingDatasetXML: WorkingDatasetXML) -> transforms.Compose:
        mean_transform_tuple = ast.literal_eval(workingDatasetXML.getTransformParameterValue('normalizeMean').text)
        std_transform_tuple = ast.literal_eval(workingDatasetXML.getTransformParameterValue('normalizeStd').text)
        compose = transforms.Compose([
            transforms.Resize(int(workingDatasetXML.getTransformParameterValue('imageResize').text))
            if 'imageResize' in workingDatasetXML.getTransformParams().keys() else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(list(mean_transform_tuple), list(std_transform_tuple))
        ])
        return compose

    def _instantiateTransformerWithoutNormalization(self, workingDatasetXML: WorkingDatasetXML) -> transforms.Compose:
        compose = transforms.Compose([
            transforms.Resize(int(workingDatasetXML.getTransformParameterValue('imageResize').text))
            if 'imageResize' in workingDatasetXML.getTransformParams().keys() else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
        ])
        return compose
