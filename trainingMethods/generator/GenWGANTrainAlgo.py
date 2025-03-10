from typing import Dict
from xml.etree import ElementTree as ET

import torch
from torch import nn as nn

from discriminatorTeachersModels.DiscriminatorTeacher import DiscriminatorTeacher
from explanation_tools.ExplanationAlgorithm import ExplanationAlgorithm
from generatorStudentsModels.GeneratorStudent import GeneratorStudent
from trainingMethods.generator.GeneratorTrainingAlgorithm import GeneratorTrainingAlgorithm
from explanation_tools.ExplanationUtils import ExplanationUtils


class GenWGANTrainAlgo(GeneratorTrainingAlgorithm):

    def trainGenerator(self, discriminatorTeacher: DiscriminatorTeacher, generatorStudent: GeneratorStudent, fake_data: torch.Tensor, local_explainable: bool,
                       trained_data_explanation: torch.Tensor, explanationAlgorithm: ExplanationAlgorithm,
                       loss: nn.Module, g_optim: torch.optim.Optimizer, real_label: float, paramsAlgo: Dict[str, ET]) -> torch.Tensor:
        BATCH_SIZE = fake_data.shape[0]

        # Reset Gradients
        generatorStudent.zero_grad()

        # Train on fake data
        prediction = discriminatorTeacher(fake_data).view(-1)

        if local_explainable:
            explanationAlgorithm.extractExplanationFeatureAttr(discriminatorTeacher, fake_data, prediction, trained_data_explanation)

        # Calculate error and back-propagation
        error = -torch.mean(prediction)
        error.backward()

        # update parameters
        g_optim.step()

        # reset explainability gradients
        ExplanationUtils.resetGradientMatrixMask()

        return error




