from typing import Dict, Any, List, Tuple
import xml.etree.ElementTree as ET


class GANLectureClassXML:

    def __init__(self, xmlRoot: ET):
        self.__id: str = ""
        self.__isTeam: bool = False
        self.__teamParams: Dict[str, ET] = {}
        self.__generatorStudentId: str = ""
        self.__discriminatorTeacherId: str = ""
        self.__workingDatasetId: str = ""
        self.__discTrainingAlgo: str = ""
        self.__genTrainingAlgo: str = ""
        self.__experimentParams: Dict[str, ET] = {}
        self.__explanationParams: Dict[str, ET] = {}
        self.__evaluationParams: Dict[str, ET] = {}
        self.__additionalMembers: List[Tuple[str, str, str]] = []
        self.__buildGeneratorComponent(xmlRoot)

    def __buildGeneratorComponent(self, root: ET) -> None:
        self.__id = root.find('id').text
        self.__generatorStudentId = root.find('generatorStudentId').text
        self.__discriminatorTeacherId = root.find('discriminatorTeacherId').text
        self.__workingDatasetId = root.find('workingDataset').text
        self.__genTrainingAlgo = root.find('generatorTrainingAlgorithm').text
        self.__discTrainingAlgo = root.find('discriminatorTrainingAlgorithm').text

        for experimentParam in root.find('experimentParams'):
            xmlTag = str(experimentParam.tag)
            self.__experimentParams[xmlTag] = experimentParam

        for explanationParam in root.find('explanationParams'):
            xmlTag = str(explanationParam.tag)
            self.__explanationParams[xmlTag] = explanationParam

        for evaluationParam in root.find('evaluationParams'):
            xmlTag = str(evaluationParam.tag)
            self.__evaluationParams[xmlTag] = evaluationParam

        if root.find('isTeam') is not None:
            self.__isTeam = bool(root.find('isTeam').text)
            if self.__isTeam:
                # for teamParam in root.find('teamParams'):
                #     xmlTag = str(teamParam.tag)
                #     self.__teamParams[xmlTag] = teamParam
                for memberParam in root.find('teamParams').find('additionalMembers').findall('member'):
                    generatorId = memberParam.find('generatorId').text
                    savingLastPath = memberParam.find('savingLastPath').text
                    if savingLastPath == None:
                        savingLastPath = ""
                    savingBestPath = memberParam.find('savingBestPath').text
                    if savingBestPath == None:
                        savingBestPath = ""
                    self.__additionalMembers.append((generatorId, savingLastPath, savingBestPath))

    def getId(self) -> str:
        return self.__id

    def isTeam(self) -> bool:
        return self.__isTeam

    def getGeneratorStudentId(self) -> str:
        return self.__generatorStudentId

    def getDiscriminatorTeacherId(self) -> str:
        return self.__discriminatorTeacherId

    def getWorkingDatasetId(self) -> str:
        return self.__workingDatasetId

    def getDiscTrainingAlgo(self) -> str:
        return self.__discTrainingAlgo

    def getGenTrainingAlgo(self) -> str:
        return self.__genTrainingAlgo

    def getExperimentParameters(self) -> Dict[str, ET]:
        return self.__experimentParams

    def getHyperParameterValue(self, hyperParameterName: str) -> ET:
        if not hyperParameterName in self.__experimentParams.keys():
            return None
        return self.__experimentParams[hyperParameterName]

    def getExplanationParameters(self) -> Dict[str, ET]:
        return self.__explanationParams

    def getExplanationParameterValue(self, explainParameterName: str) -> ET:
        if not explainParameterName in self.__explanationParams.keys():
            return None
        return self.__explanationParams[explainParameterName]

    def getEvaluationParameters(self) -> Dict[str, ET]:
        return self.__evaluationParams

    def getEvaluationParameterValue(self, evaluationParameterName: str) -> ET:
        if not evaluationParameterName in self.__evaluationParams.keys():
            return None
        return self.__evaluationParams[evaluationParameterName]

    def getTeamParameters(self) -> Dict[str, ET]:
        return self.__teamParams

    def getTeamParameterValue(self, teamParameterName: str) -> ET:
        if not teamParameterName in self.__teamParams.keys():
            return None
        return self.__teamParams[teamParameterName]

    def getAdditionalMembers(self) -> List[Tuple[str, str, str]]:
        return self.__additionalMembers


