import torch
import torch.nn as nn

from discriminatorTeachersModels.DiscriminatorTeacher import DiscriminatorTeacher
from explanation_tools.ExplanationAlgorithm import ExplanationAlgorithm
from explanation_tools.ExplanationUtils import ExplanationUtils
from generatorStudentsModels.GeneratorStudent import GeneratorStudent
from trainingMethods.generator.GeneratorTrainingAlgorithm import GeneratorTrainingAlgorithm
from utils.tensor_utils import values_target


class GenClassicTrainAlgo(GeneratorTrainingAlgorithm):



    def trainGenerator(self, discriminatorTeacher: DiscriminatorTeacher, generatorStudent: GeneratorStudent, fake_data: torch.Tensor, local_explainable: bool,
                       trained_data_explanation: torch.Tensor, explanationAlgorithm: ExplanationAlgorithm,
                       loss: nn.Module, g_optim: torch.optim.Optimizer, real_label: float) -> torch.Tensor:
        BATCH_SIZE = fake_data.shape[0]

        # Reset Gradients
        g_optim.zero_grad()

        # Train on fake data
        prediction = discriminatorTeacher(fake_data).view(-1)

        if local_explainable:
            explanationAlgorithm.extractExplanationFeatureAttr(discriminatorTeacher, fake_data, prediction, trained_data_explanation)

        # Calculate error and back-propagation
        error = loss(prediction, values_target(size=(BATCH_SIZE,), value=real_label))
        error.backward()

        # clip gradients to avoid exploding gradient problem
        nn.utils.clip_grad_norm_(generatorStudent.parameters(), 10)

        # update parameters
        g_optim.step()

        # reset explainability gradients
        ExplanationUtils.resetGradientMatrixMask()

        return error




