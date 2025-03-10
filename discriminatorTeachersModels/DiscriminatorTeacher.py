import torch.nn as nn

from typing import Dict, Any, List

from generatorStudentsModels.GeneratorStudent import GeneratorStudent


class DiscriminatorTeacher(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.__generatorStudentsList: List[GeneratorStudent] = []

    def getGeneratorStudents(self) -> List[GeneratorStudent] :
        return self.__generatorStudentsList

    def addGeneratorStudent(self, generatorStudent: GeneratorStudent) -> None:
        self.__generatorStudentsList.append(generatorStudent)

    def initialise_weights(self) -> None:
        pass

