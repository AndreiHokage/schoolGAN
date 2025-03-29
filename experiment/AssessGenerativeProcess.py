import os
import re
import subprocess
from typing import List, Dict
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import torch
import xml.etree.ElementTree as ET

from AppInstance import AppInstance
from experiment.ReviveGenModel import ReviveGenModel
from generatorStudentsFactories.GeneratorDCGANFactory import GeneratorDCGANFactory
from generatorStudentsFactories.GeneratorNNFactory import GeneratorNNFactory
from generatorStudentsFactories.GeneratorStudentAbstractFactory import GeneratorStudentAbstractFactory
from generatorStudentsFactories.GeneratorWGANFactory import GeneratorWGANFactory
from generatorStudentsModels.GeneratorStudent import GeneratorStudent
from generatorTeamFactories.GeneratorDCGANTeamFactory import GeneratorDCGANTeamFactory
from generatorTeamFactories.GeneratorTeamAbstractFactory import GeneratorTeamAbstractFactory
from generatorTeamFactories.GeneratorUNETTeamFactory import GeneratorUNETTeamFactory
from generatorTeamModels.GeneratorTeam import GeneratorTeam
from utils.tensor_utils import resetDirectory, createSavingDirectory, createSavingModelPthFile, get_device, \
    save_01Elems_as_Image, joinPth
from workingDatasets.WorkingDataset import WorkingDataset
from workingDatasetsFactories.WDatasetCIFAR10Factory import WDatasetCIFAR10Factory
from workingDatasetsFactories.WDatasetMNISTFactory import WDatasetMNISTFactory
from workingDatasetsFactories.WorkingDatasetAbstractFactory import WorkingDatasetAbstractFactory
from xmlComponents.GANLectureClassXML import GANLectureClassXML


class AssessGenerativeProcess:

    def __init__(self, schoolGANId: str, lectureClassId: str, num_samples: int):
        self.__schoolGANId: str = schoolGANId
        self.__lectureClassId: str = lectureClassId
        self.__NUM_EVAL_SAMPLES: int = num_samples
        self.__TRAINING_EPOCHS: int = 0
        self.__writer_fid = SummaryWriter(os.path.join('logs', self.__schoolGANId, self.__lectureClassId, "fid"))
        self.__generator: GeneratorStudent = None
        self.__workingDataset: WorkingDataset = None
        self.__freq: int = 3
        self.__counter_eval = 0
        self.__setupEnv()

    def __setupEnv(self) -> None:
        configFilePathXML: str = os.path.join('./config_study_group/config_school/', self.__schoolGANId + '.xml')
        root = ET.parse(configFilePathXML).getroot()
        for generatorLectureClass in root.findall('generatorLectureClass'):
            ganLectureClassXML = GANLectureClassXML(generatorLectureClass)
            lectureClassId: str = ganLectureClassXML.getId()

            if lectureClassId == self.__lectureClassId:
                self.__TRAINING_EPOCHS = int(ganLectureClassXML.getHyperParameterValue('epochs').text)
                self.__workingDataset = self.__createWorkingDataset(ganLectureClassXML.getWorkingDatasetId(), ganLectureClassXML.getExperimentParameters())

                if ganLectureClassXML.isTeam() == True:
                    self.__generator = self.__createGeneratorTeam(ganLectureClassXML, ganLectureClassXML.getExperimentParameters())
                else:
                    self.__generator = self.__createGeneratorStudent(ganLectureClassXML.getGeneratorStudentId(), ganLectureClassXML.getExperimentParameters())

    def __setupDirectoriesForSavingResults(self, epoch: int):
        self.__results_path = os.path.join('results', self.__schoolGANId, self.__lectureClassId, str(epoch))
        resetDirectory(self.__results_path)

        self.__results_path_saving_images_counterfeit = os.path.join(self.__results_path, 'counterfeitLast')
        resetDirectory(self.__results_path_saving_images_counterfeit)

        self.__results_path_saving_images_real = os.path.join(self.__results_path, 'realImages')
        resetDirectory(self.__results_path_saving_images_real)

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

        generatorTeam = generatorTeamConcreteFactory.createModel(explicitGenModelList, generatorTeamXML, experimentLevelParams).to(device=get_device())
        generatorTeam.setUpGenerationTeam()
        generatorTeam.initialise_weights()
        AppInstance().getCacheModels().addGeneratorStudent(generatorTeamId, generatorTeam)
        return generatorTeam

    def evaluateLectureClassOnEpoch(self, epoch: int) -> None:
        self.__setupDirectoriesForSavingResults(epoch)
        self.__path_saving_model_file = joinPth(self.__schoolGANId, self.__lectureClassId, epoch)

        if not os.path.exists(self.__path_saving_model_file) or self.__counter_eval < self.__freq:
            return

        self.__generator.load_state_dict(torch.load(self.__path_saving_model_file, map_location=get_device()))

        realSamples = self.__workingDataset.generateRealSamples(self.__NUM_EVAL_SAMPLES)
        save_01Elems_as_Image(realSamples, self.__results_path_saving_images_real)
        self.__generator.eval()
        counterfeitSamples = self.__generator.generateCounterfeitSamples(self.__NUM_EVAL_SAMPLES, normalizing="01")
        save_01Elems_as_Image(counterfeitSamples, self.__results_path_saving_images_counterfeit)

        command = "python -m pytorch_fid " + self.__results_path_saving_images_counterfeit + "/ " + self.__results_path_saving_images_real + "/ --device cuda:0"
        output = subprocess.check_output(command, shell=True).decode('utf-8')
        fidScore = float(re.search(r'\d+\.\d+', output).group())
        self.__writer_fid.add_scalar(f"fid_{self.__schoolGANId}_{self.__lectureClassId}", fidScore, global_step=epoch)
        self.__counter_eval = 0

    def evaluateLectureClass(self) -> None:
        for epoch in tqdm(range(50, self.__TRAINING_EPOCHS + 1)):
            self.__counter_eval = self.__counter_eval + 1
            self.evaluateLectureClassOnEpoch(epoch)
            if epoch < 128:
                command_rm_model = "rm -rf " + self.__path_saving_model_file
                os.system(command_rm_model)



