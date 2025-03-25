from typing import Dict, List

import xml.etree.ElementTree as ET

from experiment.ReviveGenModel import ReviveGenModel
from generatorStudentsModels.GeneratorStudent import GeneratorStudent
from generatorTeamModels import GeneratorTeam
from xmlComponents.GeneratorTeamXML import GeneratorTeamXML


class GeneratorTeamAbstractFactory:

    def __init__(self):
        pass

    def createModel(self, reviveGenModelList: List[ReviveGenModel], generatorTeamXML: GeneratorTeamXML, experimentLevelParams: Dict[str, ET]) -> GeneratorTeam:
        generatorStudent = self._instantiateModel(reviveGenModelList, generatorTeamXML, experimentLevelParams)
        return generatorStudent

    def _instantiateModel(self, reviveGenModelList: List[ReviveGenModel], generatorTeamXML: GeneratorTeamXML, experimentLevelParams: Dict[str, ET]) -> GeneratorTeam:
        pass


