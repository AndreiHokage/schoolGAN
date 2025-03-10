from typing import Tuple, Dict

import torch
import xml.etree.ElementTree as ET
from torch import nn as nn


from discriminatorTeachersModels.DiscriminatorTeacher import DiscriminatorTeacher
from trainingMethods.discriminator.DiscriminatorTrainingAlgorithm import DiscriminatorTrainingAlgorithm
from utils.tensor_utils import values_target


class DiscWGANTrainAlgo(DiscriminatorTrainingAlgorithm):

    def trainDiscriminator(self, discriminatorTeacher: DiscriminatorTeacher, real_data: torch.Tensor, fake_data: torch.Tensor,
                           loss: nn.Module, d_optim: torch.optim.Optimizer, real_label: float, fake_label: float, paramsAlgo: Dict[str, ET]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        BATCH_SIZE = real_data.shape[0]

        # Reset gradients
        discriminatorTeacher.zero_grad()

        # Train on real data
        prediction_real = discriminatorTeacher(real_data).view(-1)

        # Train on Fake data
        prediction_fake = discriminatorTeacher(fake_data).view(-1)

        # Calculate loss critic
        loss_critic = -(torch.mean(prediction_real) - torch.mean(prediction_fake))

        # Propagate error
        loss_critic.backward(retain_graph=True)

        # Update weights with gradients
        d_optim.step()

        # clip critic weights between -0.01, 0.01
        WEIGHT_CLIP: float = float(paramsAlgo['WEIGHT_CLIP'].text)
        for p in discriminatorTeacher.parameters():
            p.data.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)

        return loss_critic, prediction_real, prediction_fake


