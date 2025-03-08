import os.path
import xml.etree.ElementTree as ET
from typing import List, Dict

from AppInstance import AppInstance
from discriminatorTeachersFactories.DiscriminatorNNFactory import DiscriminatorNNFactory
from discriminatorTeachersFactories.DiscriminatorTeacherAbstractFactory import DiscriminatorTeacherAbstractFactory
from discriminatorTeachersModels.DiscriminatorTeacher import DiscriminatorTeacher
from experiment.LectureGAN import LectureGAN
from explanation_tools.DeepLiftShapExplanation import DeepLiftShapExplanation
from explanation_tools.ExplanationAlgorithm import ExplanationAlgorithm
from explanation_tools.LimeExplanation import LimeExplanation
from explanation_tools.SaliencyExplanation import SaliencyExplanation
from generatorStudentsFactories.GeneratorNNFactory import GeneratorNNFactory
from generatorStudentsFactories.GeneratorStudentAbstractFactory import GeneratorStudentAbstractFactory
from generatorStudentsModels import GeneratorStudent
from trainingMethods.discriminator.DiscriminatorTrainingAlgorithm import DiscriminatorTrainingAlgorithm
from trainingMethods.discriminator.factories.DiscClassicTrainAlgoFactory import DiscClassicTrainAlgoFactory
from trainingMethods.discriminator.factories.DiscTrainAlgoAbstractFactory import DiscTrainAlgoAbstractFactory
from trainingMethods.generator.GeneratorTrainingAlgorithm import GeneratorTrainingAlgorithm
from trainingMethods.generator.factories.GenClassicTrainAlgoFactory import GenClassicTrainAlgoFactory
from trainingMethods.generator.factories.GenTrainAlgoAbstractFactory import GenTrainAlgoAbstractFactory
from utils.tensor_utils import get_device
from workingDatasets.WorkingDataset import WorkingDataset
from workingDatasetsFactories.WDatasetCIFAR10Factory import WDatasetCIFAR10Factory
from workingDatasetsFactories.WDatasetMNISTFactory import WDatasetMNISTFactory
from workingDatasetsFactories.WorkingDatasetAbstractFactory import WorkingDatasetAbstractFactory
from xmlComponents.GANLectureClassXML import GANLectureClassXML


class SchoolGAN:

    def __init__(self, schoolGANFilename):
        self.__schoolGANFilename: str = schoolGANFilename
        self.__configFilePathXML: str = os.path.join('./config_study_group/config_school/', schoolGANFilename + '.xml')
        self.__lectureClasses: List[LectureGAN] = []
        self.__setup()

    def __setup(self):
        root = ET.parse(self.__configFilePathXML).getroot()
        for generatorLectureClass in root.findall('generatorLectureClass'):
            ganLectureClassXML = GANLectureClassXML(generatorLectureClass)
            generatorStudent = self.__createGeneratorStudent(ganLectureClassXML.getGeneratorStudentId(), ganLectureClassXML.getExperimentParameters())
            discriminatorTeacher = self.__createDiscriminatorTeacher(ganLectureClassXML.getDiscriminatorTeacherId(), ganLectureClassXML.getExperimentParameters())
            workingDataset = self.__createWorkingDataset(ganLectureClassXML.getWorkingDatasetId())
            genTrainingAlgo = self.__instantiateGenTrainingAlgo(ganLectureClassXML.getGenTrainingAlgo())
            discriminatorTrainingAlgo = self.__instantiateDiscTrainingAlgo(ganLectureClassXML.getDiscTrainingAlgo())
            explanationAlgorithm = self.__createExplanationAlgorithm(ganLectureClassXML.getExplanationParameters())
            lectureClassId: str = ganLectureClassXML.getId()
            lectureGan = LectureGAN(self.__schoolGANFilename, lectureClassId, generatorStudent, discriminatorTeacher, workingDataset, genTrainingAlgo, discriminatorTrainingAlgo,
                                    explanationAlgorithm, ganLectureClassXML.getExperimentParameters(), ganLectureClassXML.getExplanationParameters(),
                                    ganLectureClassXML.getEvaluationParameters())
            self.__lectureClasses.append(lectureGan)

    def __createGeneratorStudent(self, generatorStudentId: str, experimentLevelParams: Dict[str, ET]) -> GeneratorStudent:
        generatorStudentXML = AppInstance().getCatalogueEntities().getGeneratorStudentXMLById(generatorStudentId)
        modelName = generatorStudentXML.getModelName()

        generatorStudentConcreteFactory: GeneratorStudentAbstractFactory = None
        if modelName == 'GeneratorNN':
            generatorStudentConcreteFactory = GeneratorNNFactory()
        else:
            raise Exception("No specified type for generator")

        generatorStudent = generatorStudentConcreteFactory.createModel(generatorStudentXML, experimentLevelParams)
        AppInstance().getCacheModels().addGeneratorStudent(generatorStudentId, generatorStudent)
        return generatorStudent.to(device=get_device())

    def __createDiscriminatorTeacher(self, discriminatorTeacherId: str, experimentLevelParams: Dict[str, ET]) -> DiscriminatorTeacher:
        discriminatorTeacherXml = AppInstance().getCatalogueEntities().getDiscriminatorTeacherXMLById(discriminatorTeacherId)
        modelName = discriminatorTeacherXml.getModelName()

        discriminatorTeacherConcreteFactory: DiscriminatorTeacherAbstractFactory = None
        if modelName == 'DiscriminatorNN':
            discriminatorTeacherConcreteFactory = DiscriminatorNNFactory()
        else:
            raise Exception("No specified type for discriminator")

        discriminatorTeacher = discriminatorTeacherConcreteFactory.createModel(discriminatorTeacherXml, experimentLevelParams)
        AppInstance().getCacheModels().addDiscriminatorTeacher(discriminatorTeacherId, discriminatorTeacher)
        return discriminatorTeacher.to(device=get_device())

    def __createWorkingDataset(self, workingDatasetId: str) -> WorkingDataset:
        workingDatasetXml = AppInstance().getCatalogueEntities().getWorkingDatasetById(workingDatasetId)
        datasetName = workingDatasetXml.getDatasetName()

        workingDatasetFactory: WorkingDatasetAbstractFactory = None
        if datasetName == "MNIST":
            workingDatasetFactory = WDatasetMNISTFactory()
        elif datasetName == "CIFAR10":
            workingDatasetFactory = WDatasetCIFAR10Factory()
        else:
            raise Exception("No specified type for working dataset")

        workingDataset = workingDatasetFactory.createDataset(workingDatasetXml)
        return workingDataset

    '''
    id represents the identity of the algorithm, not a definition
    '''
    def __instantiateDiscTrainingAlgo(self, discTrainingAlgoId: str) -> DiscriminatorTrainingAlgorithm:
        discTrainAlgoXML = AppInstance().getCatalogueEntities().getDiscTrainAlgoById(discTrainingAlgoId)
        algoName = discTrainAlgoXML.getAlgoName()

        discTrainAlgoAbstractFactory: DiscTrainAlgoAbstractFactory = None
        if algoName == "ClassicTraining":
            discTrainAlgoAbstractFactory = DiscClassicTrainAlgoFactory()
        else:
            raise Exception("No specified type for Disc Training Algo")

        discTrainingAlgo = discTrainAlgoAbstractFactory.createDiscTrainingAlgo(discTrainAlgoXML)
        return discTrainingAlgo

    '''
        id represents the identity of the algorithm, not a definition
        '''

    def __instantiateGenTrainingAlgo(self, genTrainingAlgoId: str) -> GeneratorTrainingAlgorithm:
        genTrainAlgoXML = AppInstance().getCatalogueEntities().getGenTrainAlgoById(genTrainingAlgoId)
        algoName = genTrainAlgoXML.getAlgoName()

        genTrainAlgoAbstractFactory: GenTrainAlgoAbstractFactory = None
        if algoName == "ClassicTraining":
            genTrainAlgoAbstractFactory = GenClassicTrainAlgoFactory()
        else:
            raise Exception("No specified type for Disc Training Algo")

        genTrainingAlgo = genTrainAlgoAbstractFactory.createGenTrainingAlgo(genTrainAlgoXML)
        return genTrainingAlgo

    def __createExplanationAlgorithm(self, explanationParams: Dict[str, ET]) -> ExplanationAlgorithm:
        explanationName = explanationParams['name'].text
        lowPredictionScore = float(explanationParams['lowPredictionScore'].text)

        explanationAlgorithm: ExplanationAlgorithm = None
        if explanationName == 'DeepLiftShap':
            explanationAlgorithm = DeepLiftShapExplanation(lowPredictionScore)
        elif explanationName == 'Saliency':
            explanationAlgorithm = SaliencyExplanation(lowPredictionScore)
        elif explanationName == 'Lime':
            explanationAlgorithm = LimeExplanation(lowPredictionScore)
        else:
            raise Exception("No specified type for Explanation Algo")

        return explanationAlgorithm

    def run(self):
        for lectureClass in self.__lectureClasses:
            lectureClass.runExperiment()
