from copy import deepcopy
from itertools import count

import torch
from torcheval.metrics import FrechetInceptionDistance
from torchvision import transforms
from torchmetrics.image.inception import InceptionScore

from generatorStudentsModels.GeneratorStudent import GeneratorStudent
from utils.tensor_utils import get_device
from workingDatasets.WorkingDataset import WorkingDataset


class EvaluateLectureGAN:

    def __computeFIDScores(self, counterfeitSamples: torch.Tensor, realSamples: torch.Tensor) -> float:
        if counterfeitSamples.shape[1] == 1:
            counterfeitSamples = transforms.Grayscale(num_output_channels=3)(counterfeitSamples)

        if realSamples.shape[1] == 1:
            realSamples = transforms.Grayscale(num_output_channels=3)(realSamples)

        fid = FrechetInceptionDistance(device=get_device())
        fid.update(counterfeitSamples, is_real=False)
        fid.update(realSamples, is_real=True)

        fid_distance_item = fid.compute().item()
        return fid_distance_item

    def __computeISScore(self, counterfeitSamples: torch.Tensor) -> float:
        inpScore = InceptionScore()
        inpScore.update(counterfeitSamples)

        is_score_item = inpScore.item()
        return is_score_item

    def evaluateFIDScores(self, numSamples: int, assessGenerator: GeneratorStudent, workingDataset: WorkingDataset) -> float:
        counterfeitSamples = assessGenerator.generateCounterfeitSamples(numSamples, normalizing="01")
        realSamples = workingDataset.generateRealSamples(numSamples)

        fid_distance_item = self.__computeFIDScores(counterfeitSamples, realSamples)
        return fid_distance_item

    def evaluateFIDScoresDataProvided(self, counterfeitSamples: torch.Tensor, realSamples: torch.Tensor) -> float:
        fid_distance_item = self.__computeFIDScores(counterfeitSamples, realSamples)
        return fid_distance_item

    def evaluateISScoreDataProvided(self, counterfeitSamples: torch.Tensor) -> float:
        is_score_item = self.__computeISScore(counterfeitSamples)
        return is_score_item

    def evaluateISScore(self, numSamples: int, assessGenerator: GeneratorStudent) -> float:
        counterfeitSamples = assessGenerator.generateCounterfeitSamples(numSamples, normalizing="01")

        is_score_item = self.__computeISScore(counterfeitSamples)
        return is_score_item