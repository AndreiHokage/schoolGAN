from typing import Dict

from generatorStudentsModels.GeneratorStudent import GeneratorStudent
from xmlComponents.GeneratorStudentXML import GeneratorStudentXML
import xml.etree.ElementTree as ET


class GeneratorStudentAbstractFactory:

    def __init__(self):
        pass

    def createModel(self, generatorStudentXML: GeneratorStudentXML, experimentLevelParams: Dict[str, ET]) -> GeneratorStudent:
        generatorStudent = self._instantiateModel(generatorStudentXML, experimentLevelParams)
        return generatorStudent

    def _instantiateModel(self, generatorStudentXML: GeneratorStudentXML, experimentLevelParams: Dict[str, ET]) -> GeneratorStudent:
        pass