from typing import List, Dict
from xml.etree import ElementTree as ET

from experiment.ReviveGenModel import ReviveGenModel
from generatorTeamFactories.GeneratorTeamAbstractFactory import GeneratorTeamAbstractFactory
from generatorTeamModels import GeneratorTeam
from generatorTeamModels.GeneratorDCGANTeam import GeneratorDCGANTeam
from xmlComponents.GeneratorTeamXML import GeneratorTeamXML


class GeneratorDCGANTeamFactory(GeneratorTeamAbstractFactory):

    def _instantiateModel(self, reviveGenModelList: List[ReviveGenModel], generatorTeamXML: GeneratorTeamXML, experimentLevelParams: Dict[str, ET]) -> GeneratorTeam:
        FEATURES_G = int(generatorTeamXML.getHyperParameterValue('FEATURES_G').text)
        NUM_CHANNELS_CONCAT_STUDENTS = len(reviveGenModelList) * int(experimentLevelParams["numChannels"].text)
        IMAGE_SIZE = int(experimentLevelParams["imageSize"].text)
        generatorDCGAN = GeneratorDCGANTeam(reviveGenModelList, NUM_CHANNELS_CONCAT_STUDENTS, IMAGE_SIZE, FEATURES_G)
        return generatorDCGAN



