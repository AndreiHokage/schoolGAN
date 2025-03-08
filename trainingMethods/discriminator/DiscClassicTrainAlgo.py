from typing import Tuple

import torch
from torch import nn as nn

from discriminatorTeachersModels.DiscriminatorTeacher import DiscriminatorTeacher
from trainingMethods.discriminator.DiscriminatorTrainingAlgorithm import DiscriminatorTrainingAlgorithm
from utils.tensor_utils import values_target, get_device


class DiscClassicTrainAlgo(DiscriminatorTrainingAlgorithm):

    def __init__(self):
        super(DiscriminatorTrainingAlgorithm).__init__()

    def trainDiscriminator(self, discriminatorTeacher: DiscriminatorTeacher, real_data: torch.Tensor, fake_data: torch.Tensor,
                           loss: nn.Module, d_optim: torch.optim.Optimizer, real_label: float, fake_label: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        BATCH_SIZE = real_data.shape[0]

        # Reset gradients
        d_optim.zero_grad()

        # Train on real data
        prediction_real = discriminatorTeacher(real_data).view(-1)

        # Calculate error
        error_real = loss(prediction_real, values_target(size=(BATCH_SIZE,), value=real_label))

        # Train on Fake data
        prediction_fake = discriminatorTeacher(fake_data).view(-1)

        # Calculate error
        error_fake = loss(prediction_fake, values_target(size=(BATCH_SIZE,), value=fake_label))

        # Sum up error and backpropagate
        error = (error_real + error_fake) / 2
        error.backward()

        # Update weights with gradients
        d_optim.step()

        return error, prediction_real, prediction_fake

