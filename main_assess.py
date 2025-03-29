import os
import random
import sys

import numpy as np
import torch
import xml.etree.ElementTree as ET

from AppInstance import AppInstance
from catalog.CacheModels import CacheModels
from catalog.CatalogueEntities import CatalogueEntities
from discriminatorTeachersFactories.DiscriminatorNNFactory import DiscriminatorNNFactory
from discriminatorTeachersFactories.DiscriminatorTeacherAbstractFactory import DiscriminatorTeacherAbstractFactory
from discriminatorTeachersModels.DiscriminatorTeacher import DiscriminatorTeacher
from experiment.AssessGenerativeProcess import AssessGenerativeProcess
from experiment.SchoolGAN import SchoolGAN
from generatorStudentsFactories.GeneratorNNFactory import GeneratorNNFactory
from generatorStudentsModels.GeneratorDCGAN import GeneratorDCGAN
from generatorStudentsModels.GeneratorStudent import GeneratorStudent
from generatorStudentsFactories.GeneratorStudentAbstractFactory import GeneratorStudentAbstractFactory
from generatorStudentsModels.GeneratorWGAN import GeneratorWGAN
from utils.tensor_utils import get_device, save_01Elems_as_Image
from workingDatasets.WorkingDataset import WorkingDataset
from workingDatasetsFactories.WDatasetMNISTFactory import WDatasetMNISTFactory
from workingDatasetsFactories.WorkingDatasetAbstractFactory import WorkingDatasetAbstractFactory
from xmlComponents.DiscTrainAlgoXML import DiscTrainAlgoXML
from xmlComponents.DiscriminatorTeacherXML import DiscriminatorTeacherXML
from xmlComponents.GenTrainAlgoXML import GenTrainAlgoXML
from xmlComponents.GeneratorStudentXML import GeneratorStudentXML
from xmlComponents.GeneratorTeamXML import GeneratorTeamXML
from xmlComponents.WorkingDatasetXML import WorkingDatasetXML


# catalogueEntities = CatalogueEntities()
# cacheModels = CacheModels()

def readAllGeneratorStudentsConf():
    root = ET.parse('./config_study_group/generatorStudents.xml').getroot()
    for generatorStudent in root.findall('generatorTeacher'):
        generatorStudentXML = GeneratorStudentXML(generatorStudent)
        AppInstance().getCatalogueEntities().addGeneratorStudentXML(generatorStudentXML.getId(), generatorStudentXML)

def readAllGeneratorTeamsConf():
    root = ET.parse('./config_study_group/generatorTeams.xml').getroot()
    for generatorTeam in root.findall('generatorTeam'):
        generatorTeamXML = GeneratorTeamXML(generatorTeam)
        AppInstance().getCatalogueEntities().addGeneratorTeamXML(generatorTeamXML.getId(), generatorTeamXML)

def readAllDiscriminatorTeachersConf():
    root = ET.parse('./config_study_group/discriminatorTeachers.xml').getroot()
    for discriminatorTeacher in root.findall('discriminatorTeacher'):
        discriminatorTeacherXML = DiscriminatorTeacherXML(discriminatorTeacher)
        AppInstance().getCatalogueEntities().addDiscriminatorTeacherXML(discriminatorTeacherXML.getId(), discriminatorTeacherXML)

def readAllWorkingDatasets():
    root = ET.parse("./config_study_group/workingDatasets.xml").getroot()
    for workingDataset in root.findall('workingDataset'):
        workingDatasetXML = WorkingDatasetXML(workingDataset)
        AppInstance().getCatalogueEntities().addWorkingDatasetXML(workingDatasetXML.getId(), workingDatasetXML)

def readAllDiscTrainAlgos():
    root = ET.parse("./config_study_group/traininingAlgorithms/discriminatorTrainingAlgorithms.xml").getroot()
    for discTrainingAlgorithm in root.findall('discTrainingAlgorithm'):
        discTrainAlgoXML = DiscTrainAlgoXML(discTrainingAlgorithm)
        AppInstance().getCatalogueEntities().addDiscTrainAlgoXML(discTrainAlgoXML.getId(), discTrainAlgoXML)

def readAllGenTrainAlgos():
    root = ET.parse("./config_study_group/traininingAlgorithms/generatorTrainingAlgorithms.xml").getroot()
    for genTrainingAlgorithm in root.findall('genTrainingAlgorithm'):
        genTrainingAlgorithm = GenTrainAlgoXML(genTrainingAlgorithm)
        AppInstance().getCatalogueEntities().addGenTrainAlgoXML(genTrainingAlgorithm.getId(), genTrainingAlgorithm)

'''
id = as an entity ? or as a definition ? !!! Answer: id as an entity (we have a modelName field as well)
'''
def createGeneratorStudent(generatorStudentId: str) -> GeneratorStudent:
    generatorStudentXML = AppInstance().getCatalogueEntities().getGeneratorStudentXMLById(generatorStudentId)
    modelName = generatorStudentXML.getModelName()

    generatorStudentConcreteFactory: GeneratorStudentAbstractFactory = None
    if modelName == 'GeneratorNN':
        generatorStudentConcreteFactory = GeneratorNNFactory()
    else:
        raise Exception("No specified type for generator")

    generatorStudent = generatorStudentConcreteFactory.createModel(generatorStudentXML)
    AppInstance().getCacheModels().addGeneratorStudent(generatorStudentId, generatorStudent)
    return generatorStudent

def createDiscriminatorTeacher(discriminatorTeacherId: str) -> DiscriminatorTeacher:
    discriminatorTeacherXml = AppInstance().getCatalogueEntities().getDiscriminatorTeacherXMLById(discriminatorTeacherId)
    modelName = discriminatorTeacherXml.getModelName()

    discriminatorTeacherConcreteFactory: DiscriminatorTeacherAbstractFactory = None
    if modelName == 'DiscriminatorNN':
        discriminatorTeacherConcreteFactory = DiscriminatorNNFactory()
    else:
        raise Exception("No specified type for discriminator")

    discriminatorTeacher = discriminatorTeacherConcreteFactory.createModel(discriminatorTeacherXml)
    AppInstance().getCacheModels().addDiscriminatorTeacher(discriminatorTeacherId, discriminatorTeacher)
    return discriminatorTeacher

def createWorkingDataset(workingDatasetId: str) -> WorkingDataset:
    # get the xml file dataset
    # get the name
    datasetName = "MNIST"

    workingDatasetFactory: WorkingDatasetAbstractFactory = None
    if datasetName == "MNIST":
        workingDatasetFactory = WDatasetMNISTFactory()

    workingDataset = workingDatasetFactory.createDataset()
    return workingDataset

def set_deterministic_seed():
    # Define the seed value
    seed = 42

    # Set seed for PyTorch
    torch.manual_seed(seed)

    # Set seed for CUDA (if using GPUs)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # Set seed for Python's random module
    random.seed(seed)

    # Set seed for NumPy
    np.random.seed(seed)

    # Ensure deterministic behavior for PyTorch operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def convert2Tuple(batchSize=(1,)):
    print("CONVERTCONVET")
    print(type(batchSize))
    print(batchSize[0])

def laodngModel():
    savingGen = GeneratorWGAN(100, 3, 64, 16)
    savingGen.load_state_dict(torch.load(
        "/bigdata/userhome/andrei.balanici/Project_Software_Development/schoolGAN/saving_models/generatorLectureClasses/last_Generator_Lecture_Class_1.pth",
        map_location=lambda storage, loc: storage))

    # savingGen = GeneratorDCGAN(128, 3, 64, 64)
    # savingGen.load_state_dict(torch.load(
    #     "/bigdata/userhome/andrei.balanici/Project_Software_Development/schoolGAN/saving_models/generatorLectureClassesTeam_1/last_Generator_Lecture_Class_2.pth",
    #     map_location=lambda storage, loc: storage))

    savingGen = savingGen.to(device=get_device())
    savingGen.eval()
    pathImg = os.path.join(
        "/bigdata/userhome/andrei.balanici/Project_Software_Development/schoolGAN/results/generatorLectureClasses/Generator_Lecture_Class_2",
        'counterfeitLast')
    counterfeitSamples = savingGen.generateCounterfeitSamples(100, normalizing="01")
    os.makedirs(pathImg, exist_ok=True)
    save_01Elems_as_Image(counterfeitSamples, pathImg)

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Error: the id of the config school, the id of the lecture class, the number of samples are needed to run the SchoolGAN")
        exit(1)

    set_deterministic_seed()

    configSchoolId: str = sys.argv[1]
    configLectureId: str = sys.argv[2]
    numSamples: int = int(sys.argv[3])
    print("configSchoolID: ", configSchoolId)
    print("configLectureId: ", configLectureId)
    print("numSamples: ", numSamples)
    #torch.autograd.set_detect_anomaly(True)
    readAllGeneratorStudentsConf()
    readAllGeneratorTeamsConf()
    readAllDiscriminatorTeachersConf()
    readAllWorkingDatasets()
    readAllGenTrainAlgos()
    readAllDiscTrainAlgos()
    # createGeneratorStudent('Generator_Student_1')
    #
    # discriminatorTeacher = createDiscriminatorTeacher('Discriminator_Teacher_1')
    # s1 = AppInstance()

    #workingDataset = createWorkingDataset("")
    print("f")
    print(get_device())
    print(torch.__version__)

    # images = np.random.rand(3, 4, 4, 3)
    # images_transposed = np.transpose(images, (0, 3, 1, 2))
    # for i in images_transposed:
    #     print(i)
    #     print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
    # print("BREAKBREAKBREAKBREAKBREAKBREAKBREAKBREAKBREAKBREAK")
    # for i in torch.Tensor(images_transposed):
    #     print(i)
    #     print("___________________________________________________")

    a = torch.Tensor([[[3, 4, 5], [4, 5, 6]]])
    print(a.shape)
    print(a)
    print(a.squeeze())
    print("___III____")
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))
    print(get_device())
    print(torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu")
    print("____END_____")

    a = torch.rand((10,1,1,1))
    b = a.view(-1)

    print("EVALUATING EVALUATING EVALUATING EVALUATING EVALUATING EVALUATING EVALUATING EVALUATING EVALUATING")
    # examples of instantiation: assessGenerativeProcess = AssessGenerativeProcess('runTeam_8_2', 'TEAM', 500)
    assessGenerativeProcess = AssessGenerativeProcess(configSchoolId, configLectureId, numSamples)
    assessGenerativeProcess.evaluateLectureClass()






