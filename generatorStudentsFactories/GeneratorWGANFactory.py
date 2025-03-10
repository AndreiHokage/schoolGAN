from typing import Dict
from xml.etree import ElementTree as ET

from generatorStudentsFactories.GeneratorStudentAbstractFactory import GeneratorStudentAbstractFactory
from generatorStudentsModels.GeneratorStudent import GeneratorStudent
from generatorStudentsModels.GeneratorWGAN import GeneratorWGAN
from xmlComponents.GeneratorStudentXML import GeneratorStudentXML


class GeneratorWGANFactory(GeneratorStudentAbstractFactory):

    def _instantiateModel(self, generatorStudentXML: GeneratorStudentXML, experimentLevelParams: Dict[str, ET]) -> GeneratorStudent:
        NOISE_DIM = int(generatorStudentXML.getHyperParameterValue('NOISE_DIM').text)
        FEATURES_G = int(generatorStudentXML.getHyperParameterValue('FEATURES_G').text)
        NUM_CHANNELS = int(experimentLevelParams["numChannels"].text)
        IMAGE_SIZE = int(experimentLevelParams["imageSize"].text)
        generatorDCGAN = GeneratorWGAN(NOISE_DIM, NUM_CHANNELS, IMAGE_SIZE, FEATURES_G)
        return generatorDCGAN

