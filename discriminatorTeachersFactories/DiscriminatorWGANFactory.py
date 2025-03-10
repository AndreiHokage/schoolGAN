from typing import Dict
from xml.etree import ElementTree as ET

from discriminatorTeachersFactories.DiscriminatorTeacherAbstractFactory import DiscriminatorTeacherAbstractFactory
from discriminatorTeachersModels.DiscriminatorTeacher import DiscriminatorTeacher
from discriminatorTeachersModels.DiscriminatorWGAN import DiscriminatorWGAN
from xmlComponents.DiscriminatorTeacherXML import DiscriminatorTeacherXML


class DiscriminatorWGANFactory(DiscriminatorTeacherAbstractFactory):

    def _instantiateModel(self, discriminatorTeacherXML: DiscriminatorTeacherXML, experimentLevelParams: Dict[str, ET]) -> DiscriminatorTeacher:
        FEATURES_D = int(discriminatorTeacherXML.getHyperParameterValue('FEATURES_D').text)
        NUM_CHANNELS = int(experimentLevelParams["numChannels"].text)
        IMAGE_SIZE = int(experimentLevelParams["imageSize"].text)
        discriminatorDCGAN: DiscriminatorWGAN = DiscriminatorWGAN(NUM_CHANNELS, IMAGE_SIZE, FEATURES_D)
        return discriminatorDCGAN



