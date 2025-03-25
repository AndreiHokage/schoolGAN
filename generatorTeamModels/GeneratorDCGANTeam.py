import math

import torch
import  torch.nn as nn
from typing import List


from experiment.ReviveGenModel import ReviveGenModel
from generatorTeamModels.GeneratorTeam import GeneratorTeam
from utils.tensor_utils import get_device


class GeneratorDCGANTeam(GeneratorTeam):

    def __init__(self, reviveGenModelList: List[ReviveGenModel], NUM_CHANNELS: int, IMAGE_SIZE: int, FEATURES_G: int):
        super().__init__(reviveGenModelList)
        self.__num_channels: int = NUM_CHANNELS
        self.__out_channels: int = 3
        self.__image_size: int = IMAGE_SIZE
        self.__z_dim = self.__num_channels * self.__image_size * self.__image_size
        self.__features_g = FEATURES_G

        num_blocks: int = int(math.log2(self.__image_size)) - 2
        self.__expo_in: int = 1
        self.__expo_out: int = 2 ** num_blocks

        self.__gen_bkb = nn.Sequential(
            # Input: N x NOISE_DIM x 1 x 1
            # self._block(self.__z_dim, self.__features_g * 16, 4, 1, 0),  # img: 4x4
            # self._block(self.__features_g * 16, self.__features_g * 8, 4, 2, 1),  # img: 8x8
            # self._block(self.__features_g * 8, self.__features_g * 4, 4, 2, 1),  # img: 16x16
            # self._block(self.__features_g * 4, self.__features_g * 2, 4, 2, 1),  # img: 32x32
        )

        self.__gen_bkb.append(self._block(self.__z_dim * self.__expo_in, self.__features_g * self.__expo_out, 4, 1, 0))
        while num_blocks > 1:
            self.__expo_in = self.__expo_out
            self.__expo_out //= 2
            self.__gen_bkb.append(
                self._block(self.__features_g * self.__expo_in, self.__features_g * self.__expo_out, 4, 2, 1))
            num_blocks -= 1

        self.__out = nn.Sequential(
            nn.ConvTranspose2d(
                self.__features_g * 2, self.__out_channels, kernel_size=4, stride=2, padding=1
            ),
            # Output: N x channels_img x self.__img_size (64) x self.__img_size(64)
            nn.Tanh(),
        )

    def _block(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int) -> nn.Module:
        return nn.Sequential(
                nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False
            ),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        # (N, 6, 64, 64) -> 24576
        x = x.view(x.shape[0], -1).unsqueeze(-1).unsqueeze(-1)
        x = self.__gen_bkb(x)
        x = self.__out(x) # already are in (N x channels_img x self.__img_size (64) x self.__img_size(64))
        return x

    def generateCounterfeitSamples(self, numSamples=1, normalizing=None) -> torch.Tensor:
        noise_input = super().generateNoise(numSamples).to(device=get_device())
        counterfeit_images = self.forward(noise_input).to(device=get_device())
        if normalizing == "01":
            counterfeit_images = (counterfeit_images + 1) / 2
        elif normalizing == "0255":
            counterfeit_images = ((counterfeit_images + 1) / 2) * 255
            counterfeit_images = counterfeit_images.to(torch.uint8)
        return counterfeit_images

    """
    Convert (self.__num_channels, self.__image_size, self.__image_size) Tensor -> self.__out layer's shape
    """
    def convertExplanationGradientsToOutputLayerShape(self, gradient_matrix_mask: torch.Tensor) -> torch.Tensor:
        shaped_gradient_matrix_mask = gradient_matrix_mask.view(gradient_matrix_mask.size(0), *(self.__out_channels, self.__image_size, self.__image_size))
        return shaped_gradient_matrix_mask

    def getOutLayer(self) -> nn.Module:
        return self.__out

    def initialise_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)

