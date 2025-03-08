from typing import List

import numpy as np
import torch
import torch.nn as nn

from discriminatorTeachersModels.DiscriminatorTeacher import DiscriminatorTeacher
from explanation_tools.ExplanationLog import ExplanationLog
from explanation_tools.ExplanationUtils import ExplanationUtils
from utils.tensor_utils import values_target, get_device, normalize_tensor
from numpy import ndarray


class ExplanationAlgorithm:

    def __init__(self, low_prediction_threshold: float):
        # assigns values to each feature of an input that measure how important that feature is to the prediction
        # we penalize more the features that have major contribution in leading the image to be classified as a fake one
        # we focus the gradient on the most important features that lead the classification to fake detection and limiting the gradient on the less important ones
        self._gradient_matrix_mask: torch.Tensor = torch.Tensor()
        self.__LOW_PREDICTION_THRESHOLD = low_prediction_threshold

        self._explanationLog: ExplanationLog = None
        self._worst_counterfeit_entity_idx: int = None

    def extractExplanationFeatureAttr(self, discriminator: DiscriminatorTeacher, generated_data: torch.Tensor, prediction: torch.Tensor,
                                  training_data: torch.Tensor) -> None:
        self._gradient_matrix_mask = values_target(size=generated_data.size(), value=1.0).to(device=get_device())

        # count just for values with low prediction
        mask = (prediction < self.__LOW_PREDICTION_THRESHOLD).view(-1) # filter the SAMPLES that have been CLASSIFIED as FALSE. Now we measure which features contributed a lot to this prediction using an XAI system
        indices_forward_expl = (mask.nonzero(as_tuple=False)).detach().cpu().numpy().flatten().tolist()
        self._pickEntityForXAIVisualisation(indices_forward_expl, prediction)

        data_forward_expl = generated_data[indices_forward_expl, :]
        self._computeAttributions(discriminator, data_forward_expl, indices_forward_expl, training_data)
        self._gradient_matrix_mask = normalize_tensor(self._gradient_matrix_mask)

        # set the current gradient_matrix_mask as a global variable in order to be accessible from hook
        ExplanationUtils.setGradientMatrixMask(self._gradient_matrix_mask)

    def _computeAttributions(self, discriminator: DiscriminatorTeacher, data_forward_expl: torch.Tensor, indices_forward_expl: List, training_data: torch.Tensor) -> None:
        pass

    def getGradientMatrixMask(self) -> torch.Tensor:
        return self._gradient_matrix_mask

    def setExplanationLog(self, explanationLog: ExplanationLog) -> None:
        self._explanationLog = explanationLog

    """
    image: ndarray with shape (H, W, C)
    mask: ndarray with shape (H, W)
    """
    def visualiseWorstEntityThroughXAI(self, image: ndarray, mask: ndarray) -> None:
        pass

    def _pickEntityForXAIVisualisation(self, indices_forward_expl: List, prediction: torch.Tensor) -> None:
        if len(indices_forward_expl) == 0:
            return

        self._worst_counterfeit_entity_idx = indices_forward_expl[0]
        mini = prediction[self._worst_counterfeit_entity_idx]
        for index_expl in indices_forward_expl:
            if prediction[index_expl] < mini:
                mini = prediction[index_expl]
                self._worst_counterfeit_entity_idx = index_expl



