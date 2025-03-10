import torch
import torch.nn as nn

from explanation_tools.ExplanationUtils import ExplanationUtils
from generatorStudentsModels.GeneratorStudent import GeneratorStudent
from utils.tensor_utils import get_device


class GeneratorNN(GeneratorStudent):

    def __init__(self, NOISE_DIM: int, NUM_CHANNELS:int, IMAGE_SIZE: int):
        super(GeneratorNN, self).__init__()
        self.__z_dim: int = NOISE_DIM
        self.__num_channels: int = NUM_CHANNELS
        self.__image_size: int = IMAGE_SIZE

        self.__gen_bkb = nn.Sequential(
            nn.Linear(self.__z_dim, 256),
            nn.LeakyReLU(0.01),
        )

        self.__out = nn.Sequential(
            nn.Linear(256, self.__num_channels * self.__image_size * self.__image_size),
            nn.Tanh(),  # normalize inputs to [-1, 1] so make outputs [-1, 1]
        )

    def forward(self, x):
        x = self.__gen_bkb(x)
        x = self.__out(x)
        x = x.view(x.size(0), *(self.__num_channels, self.__image_size, self.__image_size))
        return x

    def generateNoise(self, batch_size=1) -> torch.Tensor:
        noise = torch.randn((batch_size, self.__z_dim))
        return noise

    def generateCounterfeitSamples(self, numSamples=1, normalizing=None) -> torch.Tensor:
        noise_input = self.generateNoise(numSamples).to(device=get_device())
        counterfeit_images = self.forward(noise_input).to(device=get_device())
        if normalizing == "01":
            counterfeit_images = (counterfeit_images + 1) / 2
        elif normalizing == "0255":
            counterfeit_images = ((counterfeit_images + 1) / 2) * 255
            counterfeit_images = counterfeit_images.to(torch.uint8)
        return counterfeit_images

    def convertExplanationGradientsToOutputLayerShape(self, gradient_matrix_mask: torch.Tensor) -> torch.Tensor:
        # shape: (_, self.__num_channels, self.__image_size, self.__image_size) from ExplanationAlgorithm initialisation
        shaped_gradient_matrix_mask = gradient_matrix_mask.view(gradient_matrix_mask.size(0), -1)
        return shaped_gradient_matrix_mask

    def getOutLayer(self) -> nn.Module:
        return self.__out

    def initialise_weights(self) -> None:
        return




