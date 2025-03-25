import torch

from generatorStudentsModels.GeneratorStudent import GeneratorStudent
from utils.tensor_utils import get_device


class ReviveGenModel:

    def __init__(self, generatorStudent: GeneratorStudent, savingLastModelPath: str, savingBestModelPath: str):
        self.__generatorStudent: GeneratorStudent = generatorStudent
        self.__savingLastModelPath: str = savingLastModelPath
        self.__savingBestModelPath: str = savingBestModelPath

    def reviveLastGeneratorStudent(self) -> GeneratorStudent:
        self.__generatorStudent.load_state_dict(torch.load(self.__savingLastModelPath, map_location=get_device()))
        return self.__generatorStudent


