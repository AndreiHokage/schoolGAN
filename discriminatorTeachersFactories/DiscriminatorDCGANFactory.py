from typing import Dict
from xml.etree import ElementTree as ET

from discriminatorTeachersFactories.DiscriminatorTeacherAbstractFactory import DiscriminatorTeacherAbstractFactory
from discriminatorTeachersModels.DiscriminatorDCGAN import DiscriminatorDCGAN
from discriminatorTeachersModels.DiscriminatorTeacher import DiscriminatorTeacher
from xmlComponents.DiscriminatorTeacherXML import DiscriminatorTeacherXML


class DiscriminatorDCGANFactory(DiscriminatorTeacherAbstractFactory):

    def _instantiateModel(self, discriminatorTeacherXML: DiscriminatorTeacherXML, experimentLevelParams: Dict[str, ET]) -> DiscriminatorTeacher:
        FEATURES_D = int(discriminatorTeacherXML.getHyperParameterValue('FEATURES_D').text)
        NUM_CHANNELS = int(experimentLevelParams["numChannels"].text)
        IMAGE_SIZE = int(experimentLevelParams["imageSize"].text)
        discriminatorDCGAN: DiscriminatorDCGAN = DiscriminatorDCGAN(NUM_CHANNELS, IMAGE_SIZE, FEATURES_D)
        return discriminatorDCGAN

