from typing import List

import torch
from torch import nn as nn

from experiment.ReviveGenModel import ReviveGenModel
from generatorStudentsModels.GeneratorStudent import GeneratorStudent
from utils.tensor_utils import get_device


class GeneratorTeam(GeneratorStudent):

    def __init__(self, reviveGenModelList: List[ReviveGenModel]):
        super().__init__()
        self.__generatorStudents: List[GeneratorStudent] = []
        self.__reviveGenModelList = reviveGenModelList
        self.__runSetUp: bool = False

    def setUpGenerationTeam(self):
        self.__runSetUp = True
        for reviveGenModel in self.__reviveGenModelList:
            self.__generatorStudents.append(reviveGenModel.reviveLastGeneratorStudent())
        self.__setInferenceModeEnvironment()

    def __setInferenceModeEnvironment(self) -> None:
        for generatorStudent in self.__generatorStudents:
            # set each generatorStudent on inference mode
            generatorStudent.eval()

    def isSetUp(self) -> bool:
        return self.__runSetUp

    '''
    Generates counterfeit samples from multiple approximations of the real distribution of the actual working dataset.
    The counterfeit samples act as an input noise for a generative model like the classical feed input noise
    Tensor Shape: (self.__BTCH_SIZE, C, H, W)
    '''
    def generateNoise(self, batch_size=1) -> torch.Tensor:
        listFakeData = []
        for generatorStudent in self.__generatorStudents:
            # generate fake data from a counterfeit distribution that approximates the real one
            # generate fake data
            noise = generatorStudent.generateNoise(batch_size=batch_size).to(device=get_device())
            fake_data = generatorStudent(noise).detach().to(device=get_device())

            listFakeData.append(fake_data)

        counterfeitSamplesTeam = torch.cat(listFakeData, dim=1).to(device=get_device()) # (batch_size, sigma(C_i), H, W)
        return counterfeitSamplesTeam




