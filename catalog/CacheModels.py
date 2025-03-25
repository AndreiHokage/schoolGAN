import torch.nn as nn

from typing import Dict, Any

from discriminatorTeachersModels.DiscriminatorTeacher import DiscriminatorTeacher
from generatorStudentsModels.GeneratorStudent import GeneratorStudent


class CacheModels:

    def __init__(self):
        self.__lookupCacheGeneratorStudents: Dict[str, GeneratorStudent] = {}
        self.__lookupCacheDiscriminatorTeachers: Dict[str, DiscriminatorTeacher] = {}

    def addGeneratorStudent(self, modelId: str, model: GeneratorStudent) -> None:
        pass
        #self.__lookupCacheGeneratorStudents[modelId] = model # makes an assigment, not a copy !!!!!!

    def getGeneratorStudent(self, modelId: str) -> GeneratorStudent:
        return self.__lookupCacheGeneratorStudents[modelId]

    def addDiscriminatorTeacher(self, modelId: str, model: DiscriminatorTeacher) -> None:
        pass
        #self.__lookupCacheDiscriminatorTeachers[modelId] = model

    def getDiscriminatorTeacher(self, modelId: str) -> DiscriminatorTeacher:
        return self.__lookupCacheDiscriminatorTeachers[modelId]


