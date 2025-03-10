from typing import Tuple, Dict

import torch
import torch.nn as nn
import xml.etree.ElementTree as ET

from discriminatorTeachersModels.DiscriminatorTeacher import DiscriminatorTeacher
from explanation_tools.ExplanationAlgorithm import ExplanationAlgorithm
from generatorStudentsModels.GeneratorStudent import GeneratorStudent


class GeneratorTrainingAlgorithm:

    def trainGenerator(self, discriminatorTeacher: DiscriminatorTeacher, generatorStudent: GeneratorStudent, fake_data: torch.Tensor, local_explainable: bool,
                       trained_data_explanation: torch.Tensor, explanationAlgorithm: ExplanationAlgorithm,
                       loss: nn.Module, g_optim: torch.optim.Optimizer, real_label: float, paramsAlgo: Dict[str, ET]) -> torch.Tensor:
        pass

