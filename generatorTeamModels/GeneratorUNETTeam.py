import math
from pyexpat import features
from typing import List

import torch
import torch.nn as nn

from experiment.ReviveGenModel import ReviveGenModel
from generatorStudentsModels.GeneratorStudent import GeneratorStudent
from generatorTeamModels.GeneratorTeam import GeneratorTeam
from utils.tensor_utils import get_device, device


class GeneratorUNETTeam(GeneratorTeam):

    def __init__(self, reviveGenModelList: List[ReviveGenModel], NUM_CHANNELS: int, IMAGE_SIZE: int, FEATURES_G: int):
        super().__init__(reviveGenModelList)
        self.__num_channels: int = NUM_CHANNELS
        self.__out_channels: int = 3
        self.__image_size: int = IMAGE_SIZE
        self.__features_g = FEATURES_G

        num_blocks = int(math.log2(self.__image_size)) - 1
        self.__down_blocks_list = []
        self.__initial_down = nn.Sequential(
            nn.Conv2d(self.__num_channels, self.__features_g, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )

        expo_in = 1
        expo_out = 2
        while num_blocks > 1:
            self.__down_blocks_list.append(Block(self.__features_g * expo_in, self.__features_g * expo_out, down=True, act="leaky", use_dropout=False).to(device=get_device()))
            expo_in = expo_out
            expo_out = expo_in * 2
            num_blocks = num_blocks - 1

        # Output: N x features_g * expo_in x 2 x 2
        self.__bottleneck = nn.Sequential(
            nn.Conv2d(self.__features_g * expo_in, self.__features_g * expo_in, 4, 2, 1),
            nn.ReLU()
        )
        self.__up1 = Block(self.__features_g * expo_in, self.__features_g * expo_in, down=False, act="relu", use_dropout=True)

        # Output: N x features_g * expo_in x 1 x 1
        expo_out = expo_in // 2
        self.__up_blocks_list = []
        cnt = 0
        N_ups = len(self.__down_blocks_list)
        while cnt < N_ups:
            use_dropout = True
            if cnt >= N_ups // 2:
                use_dropout = False
            # expo_in * 2 (we get expo_in features from the corresponding down block)
            self.__up_blocks_list.append(Block(self.__features_g * expo_in * 2, self.__features_g * expo_out, down=False, act="relu", use_dropout=use_dropout).to(device=get_device()))
            expo_in = expo_out
            expo_out //= 2
            cnt = cnt + 1

        self.__final_up = nn.Sequential(
            nn.ConvTranspose2d(self.__features_g * expo_in * 2, self.__out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        initial_down = self.__initial_down(x)

        down_results_list = [initial_down]
        for down_module_block in self.__down_blocks_list:
            local_result = down_module_block(down_results_list[-1])
            down_results_list.append(local_result)

        bottleneck = self.__bottleneck(down_results_list[-1])

        up_image = self.__up1(bottleneck)
        cnt = len(down_results_list) - 1
        for up_module_block in self.__up_blocks_list:
            up_image = up_module_block(torch.cat([up_image, down_results_list[cnt]], 1))
            cnt = cnt - 1

        # it should as now cnt = 0, thus initial_down = down_results_list[cnt]
        final_image = self.__final_up(torch.cat([up_image, initial_down], 1))
        return final_image

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
        return self.__final_up

    def initialise_weights(self) -> None:
        return

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False):
        super(Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode="reflect")
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
        )

        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
        self.down = down

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x



