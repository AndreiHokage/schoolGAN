import xml.etree.ElementTree as ET
from typing import Dict


class WorkingDatasetXML:

    def __init__(self, xmlRoot: ET):
        self.__id: str = ""
        self.__datasetName: str = ""
        self.__storage_path: str = ""
        self.__limitRawSize: int = -1
        self.__transformParams: Dict[str, ET] = {}
        self.__buildGeneratorComponent(xmlRoot)

    def __buildGeneratorComponent(self, root: ET) -> None:
        self.__id = root.find("id").text
        self.__datasetName = root.find("datasetName").text
        self.__storage_path = root.find("storagePath").text
        self.__limitRawSize = int(root.find("limitRawSize").text)

        for transformParam in root.find('transform'):
            keyTag = str(transformParam.tag)
            self.__transformParams[keyTag] = transformParam

    def getId(self) -> str:
        return self.__id

    def getDatasetName(self) -> str:
        return self.__datasetName

    def getStoragePath(self) -> str:
        return self.__storage_path

    def getLimitRawSize(self) -> int:
        return self.__limitRawSize

    def getTransformParams(self) -> Dict[str, ET]:
        return self.__transformParams

    def getTransformParameterValue(self, transformParameterName: str) -> ET:
        if not transformParameterName in self.__transformParams.keys():
            return None
        return self.__transformParams[transformParameterName]



