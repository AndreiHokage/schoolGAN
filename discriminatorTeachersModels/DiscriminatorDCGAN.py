import math

import torch.nn as nn


from discriminatorTeachersModels.DiscriminatorTeacher import DiscriminatorTeacher


class DiscriminatorDCGAN(DiscriminatorTeacher):

    def __init__(self, NUM_CHANNELS: int, IMAGE_SIZE: int, FEATURES_D: int):
        super(DiscriminatorDCGAN, self).__init__()
        self.__num_channels: int = NUM_CHANNELS
        self.__image_size: int = IMAGE_SIZE
        self.__features_d = FEATURES_D

        self.__disc = nn.Sequential(
            # input: N x self.__channels_img x self.__image_size x self.__image_size
            nn.Conv2d(self.__num_channels, self.__features_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )

        num_blocks: int = int(math.log2(self.__image_size)) - 1 - 2
        expo_in: int = 1
        expo_out: int = 2
        while num_blocks > 0:
            self.__disc.append(self._block(self.__features_d * expo_in, self.__features_d * expo_out, 4, 2, 1))
            expo_in = expo_out
            expo_out *= 2
            num_blocks -= 1

        # After all _block img output is 4x4 (Conv2 below makes into 1x1)
        self.__disc.append(nn.Conv2d(self.__features_d * expo_in, 1, kernel_size=4, stride=2, padding=0))
        self.__disc.append(nn.Sigmoid())


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
            # nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.__disc(x)

    def initialise_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)

