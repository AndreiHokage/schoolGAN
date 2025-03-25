import xml.etree.ElementTree as ET
from typing import Dict

from AppInstance import AppInstance
from discriminatorTeachersModels.DiscriminatorTeacher import DiscriminatorTeacher
from generatorStudentsModels.GeneratorStudent import GeneratorStudent
from xmlComponents.DiscriminatorTeacherXML import DiscriminatorTeacherXML


class DiscriminatorTeacherAbstractFactory:

    def __init__(self):
        pass

    def createModel(self, discriminatorTeacherXML: DiscriminatorTeacherXML, experimentLevelParams: Dict[str, ET]) -> DiscriminatorTeacher:
        discriminatorTeacher = self._instantiateModel(discriminatorTeacherXML, experimentLevelParams)
        # for generatorStudentId in discriminatorTeacherXML.getGeneratorStudentsIds():
        #     generatorStudent = AppInstance().getCacheModels().getGeneratorStudent(generatorStudentId) # return the same reference, not a copy
        #     discriminatorTeacher.addGeneratorStudent(generatorStudent)


        # if id(discriminatorTeacher.getGeneratorStudents()[0]) == id(AppInstance().getCacheModels().getGeneratorStudent('Generator_Student_1')):
        #     print("BALALALLALA")
        # else:
        #     print("RRRRRRRRRRRRRRRRR")

        return discriminatorTeacher

    def _instantiateModel(self, discriminatorTeacherXML: DiscriminatorTeacherXML, experimentLevelParams: Dict[str, ET]) -> DiscriminatorTeacher:
        pass




