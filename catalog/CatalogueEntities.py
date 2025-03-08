from typing import Dict, Any, List

from xmlComponents.DiscTrainAlgoXML import DiscTrainAlgoXML
from xmlComponents.DiscriminatorTeacherXML import DiscriminatorTeacherXML
from xmlComponents.GenTrainAlgoXML import GenTrainAlgoXML
from xmlComponents.GeneratorStudentXML import GeneratorStudentXML
from xmlComponents.WorkingDatasetXML import WorkingDatasetXML


class CatalogueEntities:

    def __init__(self):
        self.__generatorsStudentsXMLCatalog: Dict[str, GeneratorStudentXML] = {}
        self.__discriminatorTeachersXMLCatalog: Dict[str, DiscriminatorTeacherXML] = {}
        self.__workingDatasetsXMLCatalog: Dict[str, WorkingDatasetXML] = {}
        self.__discTrainAlgosCatalog: Dict[str, DiscTrainAlgoXML] = {}
        self.__genTrainAlgosCatalog: Dict[str, GenTrainAlgoXML] = {}

    def getGeneratorsStudentsXMLCatalog(self) -> Dict[str, GeneratorStudentXML]:
        return self.__generatorsStudentsXMLCatalog

    def getDiscriminatorTeachersXMLCatalog(self) -> Dict[str, DiscriminatorTeacherXML]:
        return self.__discriminatorTeachersXMLCatalog

    def getGeneratorStudentXMLById(self, generatorStudentId: str) -> GeneratorStudentXML:
        return self.__generatorsStudentsXMLCatalog[generatorStudentId]

    def getDiscriminatorTeacherXMLById(self, discriminatorTeacherId: str) -> DiscriminatorTeacherXML:
        return self.__discriminatorTeachersXMLCatalog[discriminatorTeacherId]

    def addGeneratorStudentXML(self, generatorStudentId: str, generatorStudentXML: GeneratorStudentXML) -> None:
        self.__generatorsStudentsXMLCatalog[generatorStudentId] = generatorStudentXML

    def addDiscriminatorTeacherXML(self, discriminatorTeacherId: str, discriminatorTeacherXML: DiscriminatorTeacherXML) -> None:
        self.__discriminatorTeachersXMLCatalog[discriminatorTeacherId] = discriminatorTeacherXML

    def getWorkingDatasetsXMLCatalogue(self) -> Dict[str, WorkingDatasetXML]:
        return self.__workingDatasetsXMLCatalog

    def getWorkingDatasetById(self, workingDatasetId: str) -> WorkingDatasetXML:
        return self.__workingDatasetsXMLCatalog[workingDatasetId]

    def addWorkingDatasetXML(self, workingDatasetId: str, workingDatasetXML: WorkingDatasetXML) -> None:
        self.__workingDatasetsXMLCatalog[workingDatasetId] = workingDatasetXML

    def getDiscTrainAlgosXMLCatalogue(self) -> Dict[str, DiscTrainAlgoXML]:
        return self.__discTrainAlgosCatalog

    def getDiscTrainAlgoById(self, discTrainAlgoId: str) -> DiscTrainAlgoXML:
        return self.__discTrainAlgosCatalog[discTrainAlgoId]

    def addDiscTrainAlgoXML(self, discTrainAlgoId: str, discTrainAlgoXML: DiscTrainAlgoXML) -> None:
        self.__discTrainAlgosCatalog[discTrainAlgoId] = discTrainAlgoXML

    def getGenTrainAlgosXMLCatalogue(self) -> Dict[str, GenTrainAlgoXML]:
        return self.__genTrainAlgosCatalog

    def getGenTrainAlgoById(self, genTrainAlgoId: str) -> GenTrainAlgoXML:
        return self.__genTrainAlgosCatalog[genTrainAlgoId]

    def addGenTrainAlgoXML(self, genTrainAlgoId: str, genTrainAlgoXML: GenTrainAlgoXML) -> None:
        self.__genTrainAlgosCatalog[genTrainAlgoId] = genTrainAlgoXML




