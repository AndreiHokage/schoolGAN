from typing import Tuple

import torch
import torch.nn as nn
from discriminatorTeachersModels.DiscriminatorTeacher import DiscriminatorTeacher


class DiscriminatorTrainingAlgorithm:

    def trainDiscriminator(self, discriminatorTeacher: DiscriminatorTeacher, real_data: torch.Tensor, fake_data: torch.Tensor,
                           loss: nn.Module, d_optim: torch.optim.Optimizer, real_label: float, fake_label: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass





