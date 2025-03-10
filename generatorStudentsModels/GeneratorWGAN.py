from torch import nn as nn

from generatorStudentsModels.GeneratorDCGAN import GeneratorDCGAN


class GeneratorWGAN(GeneratorDCGAN):

    def __init__(self, NOISE_DIM: int, NUM_CHANNELS: int, IMAGE_SIZE: int, FEATURES_G: int):
        super().__init__(NOISE_DIM, NUM_CHANNELS, IMAGE_SIZE, FEATURES_G)

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
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )


