import torch.nn as nn

from discriminatorTeachersModels.DiscriminatorTeacher import DiscriminatorTeacher


class DiscriminatorNN(DiscriminatorTeacher):

    def __init__(self, IN_FEATURES: int):
        super(DiscriminatorNN, self).__init__()
        self.__in_features = IN_FEATURES

        self.disc = nn.Sequential(
            nn.Linear(self.__in_features, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.disc(x)

    def initialise_weights(self) -> None:
        return





