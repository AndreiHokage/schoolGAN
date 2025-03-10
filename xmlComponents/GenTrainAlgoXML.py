from typing import Dict, Any, List
import xml.etree.ElementTree as ET

class GenTrainAlgoXML:

    def __init__(self, xmlRoot: ET):
        self.__id: str = ""
        self.__algoName: str = ""
        self.__paramsAlgo: Dict[str, ET] = {}
        self.__buildDiscriminatorComponent(xmlRoot)

    def __buildDiscriminatorComponent(self, root: ET) -> None:
        self.__id = root.find('id').text
        self.__algoName = root.find('algoName').text

        for paramAlgo in root.find('paramsAlgo'):
            xmlTag = str(paramAlgo.tag)
            self.__paramsAlgo[xmlTag] = paramAlgo

    def getId(self) -> str:
        return self.__id

    def getAlgoName(self) -> str:
        return self.__algoName

    def getParamsAlgoParameters(self) -> Dict[str, ET]:
        return self.__paramsAlgo

    def getTrainAlgoParameterValue(self, algoParameterName: str) -> ET:
        if not algoParameterName in self.__paramsAlgo.keys():
            return None
        return self.__paramsAlgo[algoParameterName]



