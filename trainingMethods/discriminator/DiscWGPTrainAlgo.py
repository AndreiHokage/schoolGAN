from typing import Dict, Tuple
from xml.etree import ElementTree as ET

import torch
from torch import nn as nn

from discriminatorTeachersModels.DiscriminatorTeacher import DiscriminatorTeacher
from trainingMethods.discriminator.DiscriminatorTrainingAlgorithm import DiscriminatorTrainingAlgorithm
from utils.tensor_utils import get_device


class DiscWGPTrainAlgo(DiscriminatorTrainingAlgorithm):

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
        LAMBDA_GP = float(paramsAlgo['LAMBDA_GP'].text)
        gp = self.__gradient_penalty(discriminatorTeacher, real_data, fake_data)
        loss_critic = (
            -(torch.mean(prediction_real) - torch.mean(prediction_fake)) + LAMBDA_GP * gp
        )

        # Propagate error
        loss_critic.backward(retain_graph=True)

        # Update weights with gradients
        d_optim.step()

        return loss_critic, prediction_real, prediction_fake

    def __gradient_penalty(self, discriminatorTeacher: DiscriminatorTeacher, real_data: torch.Tensor, fake_data: torch.Tensor):
        BATCH_SIZE, C, H, W = real_data.shape
        alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(get_device())
        interpolated_images = real_data * alpha + fake_data * (1 - alpha)
        interpolated_images.requires_grad_(True)

        # Calculate critic scores
        mixed_scores = discriminatorTeacher(interpolated_images)

        # Take the gradient of the scores with respect to the images
        # print("dsfsdfsdf ", interpolated_images.requires_grad)
        gradient = torch.autograd.grad(
            inputs=interpolated_images,
            outputs=mixed_scores,
            grad_outputs=torch.ones_like(mixed_scores),
            create_graph=True,
            retain_graph=True,
        )[0]
        gradient = gradient.view(gradient.shape[0], -1)
        gradient_norm = gradient.norm(2, dim=1)
        gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
        # print(gradient_penalty)
        return gradient_penalty



