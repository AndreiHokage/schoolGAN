from typing import Dict, Any, List
import xml.etree.ElementTree as ET

class GenTrainAlgoXML:

    def __init__(self, xmlRoot: ET):
        self.__id: str = ""
        self.__algoName: str = ""
        self.__buildDiscriminatorComponent(xmlRoot)

    def __buildDiscriminatorComponent(self, root: ET) -> None:
        self.__id = root.find('id').text
        self.__algoName = root.find('algoName').text

    def getId(self) -> str:
        return self.__id

    def getAlgoName(self) -> str:
        return self.__algoName
    



