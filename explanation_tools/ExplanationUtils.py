import os
from typing import Tuple, Any

import torch
from jinja2.nodes import Tuple
from torch import Tensor

from generatorStudentsModels.GeneratorStudent import GeneratorStudent


class ExplanationUtils:

    gradient_matrix_mask: torch.Tensor = torch.Tensor()
    weight_explanation_grad: float = 0.0
    generatorStudent: GeneratorStudent = None

    @staticmethod
    def setGeneratorStudent(generatorStudent: GeneratorStudent) -> None:
        ExplanationUtils.generatorStudent = generatorStudent

    @staticmethod
    def getGeneratorStudent() -> GeneratorStudent:
        return ExplanationUtils.generatorStudent

    @staticmethod
    def setWeightExplanationGrad(weight_explanation_grad) -> None:
        ExplanationUtils.weight_explanation_grad = weight_explanation_grad

    '''
    Not thread-safe
    '''
    @staticmethod
    def getGradientMatrixMask() -> torch.Tensor:
        return ExplanationUtils.gradient_matrix_mask

    '''
    Not thread-safe
    '''
    @staticmethod
    def setGradientMatrixMask(gradient_matrix_mask: torch.Tensor) -> None:
        ExplanationUtils.gradient_matrix_mask = gradient_matrix_mask

    '''
    Not thread-safe
    '''
    @staticmethod
    def resetGradientMatrixMask() -> None:
        ExplanationUtils.gradient_matrix_mask = torch.Tensor()

    """
    This function creates explanation hook which is run every time the backward function of the gradients is called
    :param module: the name of the layer
    :param grad_input: the gradients from the input layer
    :param grad_output: the gradients from the output layer
    :return:
    """
    @staticmethod
    def explanation_hook(module, grad_input, grad_output) -> tuple[float | Tensor | Any]:
        shaped_gradient_matrix_mask = ExplanationUtils.generatorStudent.convertExplanationGradientsToOutputLayerShape(ExplanationUtils.gradient_matrix_mask)
        new_grad = grad_input[0] + ExplanationUtils.weight_explanation_grad * (grad_input[0] * shaped_gradient_matrix_mask)
        return (new_grad, )

