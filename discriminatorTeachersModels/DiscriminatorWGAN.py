from torch import nn

from discriminatorTeachersModels.DiscriminatorDCGAN import DiscriminatorDCGAN


class DiscriminatorWGAN(DiscriminatorDCGAN):

    def __init__(self, NUM_CHANNELS: int, IMAGE_SIZE: int, FEATURES_D: int):
        super().__init__(NUM_CHANNELS, IMAGE_SIZE, FEATURES_D)

    def _block(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )


