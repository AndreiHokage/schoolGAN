import os.path
import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple

from AppInstance import AppInstance
from discriminatorTeachersFactories.DiscriminatorDCGANFactory import DiscriminatorDCGANFactory
from discriminatorTeachersFactories.DiscriminatorNNFactory import DiscriminatorNNFactory
from discriminatorTeachersFactories.DiscriminatorTeacherAbstractFactory import DiscriminatorTeacherAbstractFactory
from discriminatorTeachersFactories.DiscriminatorWGANFactory import DiscriminatorWGANFactory
from discriminatorTeachersModels.DiscriminatorTeacher import DiscriminatorTeacher
from experiment.LectureGAN import LectureGAN
from experiment.ReviveGenModel import ReviveGenModel
from explanation_tools.DeepLiftShapExplanation import DeepLiftShapExplanation
from explanation_tools.ExplanationAlgorithm import ExplanationAlgorithm
from explanation_tools.LimeExplanation import LimeExplanation
from explanation_tools.SaliencyExplanation import SaliencyExplanation
from generatorStudentsFactories.GeneratorDCGANFactory import GeneratorDCGANFactory
from generatorStudentsFactories.GeneratorNNFactory import GeneratorNNFactory
from generatorStudentsFactories.GeneratorStudentAbstractFactory import GeneratorStudentAbstractFactory
from generatorStudentsFactories.GeneratorWGANFactory import GeneratorWGANFactory
from generatorStudentsModels import GeneratorStudent
from generatorTeamFactories import GeneratorTeamAbstractFactory
from generatorTeamFactories.GeneratorDCGANTeamFactory import GeneratorDCGANTeamFactory
from generatorTeamFactories.GeneratorUNETTeamFactory import GeneratorUNETTeamFactory
from generatorTeamModels import GeneratorTeam
from trainingMethods.discriminator.DiscriminatorTrainingAlgorithm import DiscriminatorTrainingAlgorithm
from trainingMethods.discriminator.factories.DiscClassicTrainAlgoFactory import DiscClassicTrainAlgoFactory
from trainingMethods.discriminator.factories.DiscTrainAlgoAbstractFactory import DiscTrainAlgoAbstractFactory
from trainingMethods.discriminator.factories.DiscWGPTrainAlgoFactory import DiscWGPTrainAlgoFactory
from trainingMethods.discriminator.factories.DiscriminatorWGANTrainAlgoFactory import DiscriminatorWGANTrainAlgoFactory
from trainingMethods.generator.GeneratorTrainingAlgorithm import GeneratorTrainingAlgorithm
from trainingMethods.generator.factories.GenClassicTrainAlgoFactory import GenClassicTrainAlgoFactory
from trainingMethods.generator.factories.GenTrainAlgoAbstractFactory import GenTrainAlgoAbstractFactory
from trainingMethods.generator.factories.GenWGANTrainAlgoFactory import GenWGANTrainAlgoFactory
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
        self.__reviveGenModelList: List[ReviveGenModel] = []

    def run(self):
        root = ET.parse(self.__configFilePathXML).getroot()
        for generatorLectureClass in root.findall('generatorLectureClass'):
            ganLectureClassXML = GANLectureClassXML(generatorLectureClass)

            if ganLectureClassXML.isTeam() == True:
                continue

            lectureClassId: str = ganLectureClassXML.getId()
            saving_best_model_path = os.path.join('saving_models', self.__schoolGANFilename)
            saving_best_model_path_file = os.path.join(saving_best_model_path, 'best_' + lectureClassId + '.pth')
            saving_last_model_path_file = os.path.join(saving_best_model_path, 'last_' + lectureClassId + '.pth')
            if not self.__isAnExistingAndTrainedModel(saving_last_model_path_file):
                generatorStudent = self.__createGeneratorStudent(ganLectureClassXML.getGeneratorStudentId(), ganLectureClassXML.getExperimentParameters())
                discriminatorTeacher = self.__createDiscriminatorTeacher(ganLectureClassXML.getDiscriminatorTeacherId(), ganLectureClassXML.getExperimentParameters())
                workingDataset = self.__createWorkingDataset(ganLectureClassXML.getWorkingDatasetId(), ganLectureClassXML.getExperimentParameters())
                genTrainingAlgo, genParamsTrainAlgo = self.__instantiateGenTrainingAlgo(ganLectureClassXML.getGenTrainingAlgo())
                discriminatorTrainingAlgo, discParamsTrainAlgo = self.__instantiateDiscTrainingAlgo(ganLectureClassXML.getDiscTrainingAlgo())
                explanationAlgorithm = self.__createExplanationAlgorithm(ganLectureClassXML.getExplanationParameters())
                lectureGan = LectureGAN(self.__schoolGANFilename, lectureClassId, generatorStudent, discriminatorTeacher, workingDataset, genTrainingAlgo, discriminatorTrainingAlgo,
                                        explanationAlgorithm, ganLectureClassXML.getExperimentParameters(), ganLectureClassXML.getExplanationParameters(),
                                        ganLectureClassXML.getEvaluationParameters(), genParamsTrainAlgo, discParamsTrainAlgo)

                print("NEW >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> RUN: " + lectureGan.getLectureClassId())
                lectureGan.runExperiment()
                lectureGan.freeUpLectureClass()
                del lectureGan
            else:
                print("IMPORT >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> : " + saving_best_model_path_file + " ; " + saving_last_model_path_file)

            templateGeneratorStudent = self.__createGeneratorStudent(ganLectureClassXML.getGeneratorStudentId(), ganLectureClassXML.getExperimentParameters())
            self.__reviveGenModelList.append(ReviveGenModel(templateGeneratorStudent,
                                                         saving_last_model_path_file,
                                                         saving_best_model_path_file))

        for generatorLectureClass in root.findall('generatorLectureClass'):
            ganLectureClassXML = GANLectureClassXML(generatorLectureClass)
            if ganLectureClassXML.isTeam() == False:
                continue

            print("generatorLectureClass is LEADER >>>>>>>>>>>>>>>> Team Generator model run training")
            # Although it is a generatorTeam, in the lecture class xml conf file the only way we make reference to the generator is through the id.
            # The id of the generator team is the same with the id of generator student
            generatorTeam = self.__createGeneratorTeam(ganLectureClassXML, ganLectureClassXML.getExperimentParameters())
            discriminatorTeacher = self.__createDiscriminatorTeacher(ganLectureClassXML.getDiscriminatorTeacherId(), ganLectureClassXML.getExperimentParameters())
            workingDataset = self.__createWorkingDataset(ganLectureClassXML.getWorkingDatasetId(), ganLectureClassXML.getExperimentParameters())
            genTrainingAlgo, genParamsTrainAlgo = self.__instantiateGenTrainingAlgo(ganLectureClassXML.getGenTrainingAlgo())
            discriminatorTrainingAlgo, discParamsTrainAlgo = self.__instantiateDiscTrainingAlgo(ganLectureClassXML.getDiscTrainingAlgo())
            explanationAlgorithm = self.__createExplanationAlgorithm(ganLectureClassXML.getExplanationParameters())
            lectureClassId: str = ganLectureClassXML.getId()
            lectureTeamGan = LectureGAN(self.__schoolGANFilename, lectureClassId, generatorTeam, discriminatorTeacher, workingDataset, genTrainingAlgo, discriminatorTrainingAlgo,
                                    explanationAlgorithm, ganLectureClassXML.getExperimentParameters(), ganLectureClassXML.getExplanationParameters(),
                                    ganLectureClassXML.getEvaluationParameters(), genParamsTrainAlgo, discParamsTrainAlgo)

            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Run BOSSS")
            lectureTeamGan.runExperiment()

    def __isAnExistingAndTrainedModel(self, saving_last_model_path_file):
        return os.path.exists(saving_last_model_path_file)

    def __createGeneratorStudent(self, generatorStudentId: str, experimentLevelParams: Dict[str, ET]) -> GeneratorStudent:
        generatorStudentXML = AppInstance().getCatalogueEntities().getGeneratorStudentXMLById(generatorStudentId)
        modelName = generatorStudentXML.getModelName()

        generatorStudentConcreteFactory: GeneratorStudentAbstractFactory = None
        if modelName == 'GeneratorNN':
            generatorStudentConcreteFactory = GeneratorNNFactory()
        elif modelName == 'GeneratorDCGAN':
            generatorStudentConcreteFactory = GeneratorDCGANFactory()
        elif modelName == 'GeneratorWGAN':
            generatorStudentConcreteFactory = GeneratorWGANFactory()
        else:
            raise Exception("No specified type for generator")

        generatorStudent = generatorStudentConcreteFactory.createModel(generatorStudentXML, experimentLevelParams).to(device=get_device())
        generatorStudent.initialise_weights()
        AppInstance().getCacheModels().addGeneratorStudent(generatorStudentId, generatorStudent)
        return generatorStudent

    def __createGeneratorTeam(self, ganLectureClassXML: GANLectureClassXML, experimentLevelParams: Dict[str, ET]) -> GeneratorTeam:
        generatorTeamId = ganLectureClassXML.getGeneratorStudentId()
        generatorTeamXML = AppInstance().getCatalogueEntities().getGeneratorTeamXMLById(generatorTeamId)
        modelName = generatorTeamXML.getModelName()

        generatorTeamConcreteFactory: GeneratorTeamAbstractFactory = None
        if modelName == 'GeneratorUNETTeam':
            generatorTeamConcreteFactory = GeneratorUNETTeamFactory()
        elif modelName == 'GeneratorDCGANTeam':
            generatorTeamConcreteFactory = GeneratorDCGANTeamFactory()
        else:
            raise Exception("No specified type for team generator")

        explicitGenModelList: List[ReviveGenModel] = []
        for (genId, saving_last_model_path_file, saving_best_model_path_file) in ganLectureClassXML.getAdditionalMembers():
            # WARNING: Push the experimentLevelParams of the Leader Entity, not of each student generator individually
            templateGeneratorStudent = self.__createGeneratorStudent(genId, experimentLevelParams)
            explicitGenModelList.append(ReviveGenModel(templateGeneratorStudent,
                                                            saving_last_model_path_file,
                                                            saving_best_model_path_file))

        generatorTeam = generatorTeamConcreteFactory.createModel(self.__reviveGenModelList + explicitGenModelList, generatorTeamXML, experimentLevelParams).to(device=get_device())
        generatorTeam.setUpGenerationTeam()
        generatorTeam.initialise_weights()
        AppInstance().getCacheModels().addGeneratorStudent(generatorTeamId, generatorTeam)
        return generatorTeam

    def __createDiscriminatorTeacher(self, discriminatorTeacherId: str, experimentLevelParams: Dict[str, ET]) -> DiscriminatorTeacher:
        discriminatorTeacherXml = AppInstance().getCatalogueEntities().getDiscriminatorTeacherXMLById(discriminatorTeacherId)
        modelName = discriminatorTeacherXml.getModelName()

        discriminatorTeacherConcreteFactory: DiscriminatorTeacherAbstractFactory = None
        if modelName == 'DiscriminatorNN':
            discriminatorTeacherConcreteFactory = DiscriminatorNNFactory()
        elif modelName == 'DiscriminatorDCGAN':
            discriminatorTeacherConcreteFactory = DiscriminatorDCGANFactory()
        elif modelName == 'DiscriminatorWGAN':
            discriminatorTeacherConcreteFactory = DiscriminatorWGANFactory()
        else:
            raise Exception("No specified type for discriminator")

        discriminatorTeacher = discriminatorTeacherConcreteFactory.createModel(discriminatorTeacherXml, experimentLevelParams).to(device=get_device())
        discriminatorTeacher.initialise_weights()
        AppInstance().getCacheModels().addDiscriminatorTeacher(discriminatorTeacherId, discriminatorTeacher)
        return discriminatorTeacher

    def __createWorkingDataset(self, workingDatasetId: str, experimentLevelParams: Dict[str, ET]) -> WorkingDataset:
        workingDatasetXml = AppInstance().getCatalogueEntities().getWorkingDatasetById(workingDatasetId)
        datasetName = workingDatasetXml.getDatasetName()

        workingDatasetFactory: WorkingDatasetAbstractFactory = None
        if datasetName == "MNIST":
            workingDatasetFactory = WDatasetMNISTFactory()
        elif datasetName == "CIFAR10":
            workingDatasetFactory = WDatasetCIFAR10Factory()
        else:
            raise Exception("No specified type for working dataset")

        workingDataset = workingDatasetFactory.createDataset(workingDatasetXml, experimentLevelParams)
        return workingDataset

    '''
    id represents the identity of the algorithm, not a definition
    '''
    def __instantiateDiscTrainingAlgo(self, discTrainingAlgoId: str) -> Tuple[DiscriminatorTrainingAlgorithm, Dict[str, ET]]:
        discTrainAlgoXML = AppInstance().getCatalogueEntities().getDiscTrainAlgoById(discTrainingAlgoId)
        algoName = discTrainAlgoXML.getAlgoName()

        discTrainAlgoAbstractFactory: DiscTrainAlgoAbstractFactory = None
        if algoName == "ClassicTraining":
            discTrainAlgoAbstractFactory = DiscClassicTrainAlgoFactory()
        elif algoName == "WGANTraining":
            discTrainAlgoAbstractFactory = DiscriminatorWGANTrainAlgoFactory()
        elif algoName == "WGPTraining":
            discTrainAlgoAbstractFactory = DiscWGPTrainAlgoFactory()
        else:
            raise Exception("No specified type for Disc Training Algo")

        discTrainingAlgo = discTrainAlgoAbstractFactory.createDiscTrainingAlgo(discTrainAlgoXML)
        return discTrainingAlgo, discTrainAlgoXML.getParamsAlgoParameters()

    '''
        id represents the identity of the algorithm, not a definition
        '''

    def __instantiateGenTrainingAlgo(self, genTrainingAlgoId: str) -> Tuple[GeneratorTrainingAlgorithm, Dict[str, ET]]:
        genTrainAlgoXML = AppInstance().getCatalogueEntities().getGenTrainAlgoById(genTrainingAlgoId)
        algoName = genTrainAlgoXML.getAlgoName()

        genTrainAlgoAbstractFactory: GenTrainAlgoAbstractFactory = None
        if algoName == "ClassicTraining":
            genTrainAlgoAbstractFactory = GenClassicTrainAlgoFactory()
        elif algoName == "WGANTraining":
            genTrainAlgoAbstractFactory = GenWGANTrainAlgoFactory()
        else:
            raise Exception("No specified type for Disc Training Algo")

        genTrainingAlgo = genTrainAlgoAbstractFactory.createGenTrainingAlgo(genTrainAlgoXML)
        return genTrainingAlgo, genTrainAlgoXML.getParamsAlgoParameters()

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

    # def run(self):
    #     for lectureClass in self.__lectureClasses:
    #         print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> RUN: " + lectureClass.getLectureClassId())
    #         lectureClass.runExperiment()
    #         lectureClass.freeUpLectureClass()
    #
    #     print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Run BOSSS")
    #     self.__lectureTeam.runExperiment()


