import random

import numpy as np
import torch
import xml.etree.ElementTree as ET

from AppInstance import AppInstance
from catalog.CacheModels import CacheModels
from catalog.CatalogueEntities import CatalogueEntities
from discriminatorTeachersFactories.DiscriminatorNNFactory import DiscriminatorNNFactory
from discriminatorTeachersFactories.DiscriminatorTeacherAbstractFactory import DiscriminatorTeacherAbstractFactory
from discriminatorTeachersModels.DiscriminatorTeacher import DiscriminatorTeacher
from experiment.SchoolGAN import SchoolGAN
from generatorStudentsFactories.GeneratorNNFactory import GeneratorNNFactory
from generatorStudentsModels.GeneratorStudent import GeneratorStudent
from generatorStudentsFactories.GeneratorStudentAbstractFactory import GeneratorStudentAbstractFactory
from utils.tensor_utils import get_device
from workingDatasets.WorkingDataset import WorkingDataset
from workingDatasetsFactories.WDatasetMNISTFactory import WDatasetMNISTFactory
from workingDatasetsFactories.WorkingDatasetAbstractFactory import WorkingDatasetAbstractFactory
from xmlComponents.DiscTrainAlgoXML import DiscTrainAlgoXML
from xmlComponents.DiscriminatorTeacherXML import DiscriminatorTeacherXML
from xmlComponents.GenTrainAlgoXML import GenTrainAlgoXML
from xmlComponents.GeneratorStudentXML import GeneratorStudentXML
from xmlComponents.WorkingDatasetXML import WorkingDatasetXML


# catalogueEntities = CatalogueEntities()
# cacheModels = CacheModels()

def readAllGeneratorStudentsConf():
    root = ET.parse('./config_study_group/generatorStudents.xml').getroot()
    for generatorStudent in root.findall('generatorTeacher'):
        generatorStudentXML = GeneratorStudentXML(generatorStudent)
        AppInstance().getCatalogueEntities().addGeneratorStudentXML(generatorStudentXML.getId(), generatorStudentXML)

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

if __name__ == '__main__':
    set_deterministic_seed()
    #torch.autograd.set_detect_anomaly(True)
    readAllGeneratorStudentsConf()
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
    schoolGAN = SchoolGAN('generatorLectureClasses')
    schoolGAN.run()





