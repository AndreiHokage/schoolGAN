from typing import Dict, List
from xml.etree import ElementTree as ET

from experiment.ReviveGenModel import ReviveGenModel
from generatorStudentsModels.GeneratorStudent import GeneratorStudent
from generatorTeamFactories.GeneratorTeamAbstractFactory import GeneratorTeamAbstractFactory
from generatorTeamModels import GeneratorTeam
from generatorTeamModels.GeneratorUNETTeam import GeneratorUNETTeam
from xmlComponents.GeneratorTeamXML import GeneratorTeamXML


class GeneratorUNETTeamFactory(GeneratorTeamAbstractFactory):

    def _instantiateModel(self, reviveGenModelList: List[ReviveGenModel], generatorStudentXML: GeneratorTeamXML, experimentLevelParams: Dict[str, ET]) -> GeneratorTeam:
        FEATURES_G = int(generatorStudentXML.getHyperParameterValue('FEATURES_G').text)
        NUM_CHANNELS_CONCAT_STUDENTS = len(reviveGenModelList) * int(experimentLevelParams["numChannels"].text)
        IMAGE_SIZE = int(experimentLevelParams["imageSize"].text)
        generatorDCGAN = GeneratorUNETTeam(reviveGenModelList, NUM_CHANNELS_CONCAT_STUDENTS, IMAGE_SIZE, FEATURES_G)
        return generatorDCGAN


