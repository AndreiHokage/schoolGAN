import torch
import torch.nn as nn

class GeneratorStudent(nn.Module):

    def __init__(self):
        super().__init__()

    def generateNoise(self, batch_size=1) -> torch.Tensor:
        pass

    def generateCounterfeitSamples(self, numSamples=1, normalizing=None) -> torch.Tensor:
        pass

    """
    Convert (self.__num_channels, self.__image_size, self.__image_size) Tensor -> self.__out layer's shape
    """
    def convertExplanationGradientsToOutputLayerShape(self, gradient_matrix_mask: torch.Tensor) -> torch.Tensor:
        pass

    def getOutLayer(self) -> nn.Module:
        pass

    def initialise_weights(self) -> None:
        pass

