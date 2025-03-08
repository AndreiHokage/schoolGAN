from typing import Dict, Any, List
import xml.etree.ElementTree as ET


class DiscriminatorTeacherXML:

    def __init__(self, xmlRoot: ET):
        self.__id: str = ""
        self.__modelName: str = ""
        self.__hyperParameters: Dict[str, ET] = {}
        self.__generatorStudentsIds: List[str] = []
        self.__buildDiscriminatorComponent(xmlRoot)

    def __buildDiscriminatorComponent(self, root: ET) -> None:
        self.__id = root.find('id').text
        self.__modelName = root.find('modelClass').text
        for hyperParam in root.find('hyperParameters').findall('hyperParameter'):
            hyperParamName = hyperParam.find('name').text
            hyperParamValue = hyperParam.find('value')
            self.__hyperParameters[hyperParamName] = hyperParamValue

        for generatorStudent in root.find('generatorStudents').findall('generatorStudent'):
            idGeneratorStudent = generatorStudent.find('id').text
            self.__generatorStudentsIds.append(idGeneratorStudent)

    def getId(self) -> str:
        return self.__id

    def getModelName(self) -> str:
        return self.__modelName

    def getHyperParameters(self) -> Dict[str, ET]:
        return self.__hyperParameters

    def getHyperParameterValue(self, hyperParameterName: str) -> ET:
        if not hyperParameterName in self.__hyperParameters.keys():
            return None
        return self.__hyperParameters[hyperParameterName]

    def getGeneratorStudentsIds(self):
        return self.__generatorStudentsIds



