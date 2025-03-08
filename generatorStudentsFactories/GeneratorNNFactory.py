import xml.etree.ElementTree as ET
from typing import Dict

from generatorStudentsModels.GeneratorNN import GeneratorNN
from generatorStudentsModels.GeneratorStudent import GeneratorStudent
from generatorStudentsFactories.GeneratorStudentAbstractFactory import GeneratorStudentAbstractFactory
from xmlComponents.GeneratorStudentXML import GeneratorStudentXML


class GeneratorNNFactory(GeneratorStudentAbstractFactory):

    def _instantiateModel(self, generatorStudentXML: GeneratorStudentXML, experimentLevelParams: Dict[str, ET]) -> GeneratorStudent:
        NOISE_DIM = int(generatorStudentXML.getHyperParameterValue('NOISE_DIM').text)
        NUM_CHANNELS = int(experimentLevelParams["numChannels"].text)
        IMAGE_SIZE = int(experimentLevelParams["imageSize"].text)
        generatorNN = GeneratorNN(NOISE_DIM, NUM_CHANNELS, IMAGE_SIZE)
        return generatorNN