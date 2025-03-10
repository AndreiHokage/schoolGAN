from typing import Tuple

import torch
import torch.nn as nn
import xml.etree.ElementTree as ET

from discriminatorTeachersModels.DiscriminatorTeacher import DiscriminatorTeacher
from typing import Dict, Any, List


class DiscriminatorTrainingAlgorithm:

    def trainDiscriminator(self, discriminatorTeacher: DiscriminatorTeacher, real_data: torch.Tensor, fake_data: torch.Tensor,
                           loss: nn.Module, d_optim: torch.optim.Optimizer, real_label: float, fake_label: float, paramsAlgo: Dict[str, ET]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass





