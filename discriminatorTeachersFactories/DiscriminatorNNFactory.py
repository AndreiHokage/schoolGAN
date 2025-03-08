import xml.etree.ElementTree as ET
from typing import Dict

from discriminatorTeachersModels.DiscriminatorNN import DiscriminatorNN
from discriminatorTeachersModels.DiscriminatorTeacher import DiscriminatorTeacher
from discriminatorTeachersFactories.DiscriminatorTeacherAbstractFactory import DiscriminatorTeacherAbstractFactory
from xmlComponents.DiscriminatorTeacherXML import DiscriminatorTeacherXML


class DiscriminatorNNFactory(DiscriminatorTeacherAbstractFactory):

    def _instantiateModel(self, discriminatorTeacherXML: DiscriminatorTeacherXML, experimentLevelParams: Dict[str, ET]) -> DiscriminatorTeacher:
        NUM_CHANNELS = int(experimentLevelParams["numChannels"].text)
        IMAGE_SIZE = int(experimentLevelParams["imageSize"].text)
        discriminatorTeacher = DiscriminatorNN(NUM_CHANNELS * IMAGE_SIZE * IMAGE_SIZE)
        return discriminatorTeacher

