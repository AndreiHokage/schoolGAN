import os.path
from copy import deepcopy
from locale import normalize
from typing import Dict, Any, List

import torch.optim
import torch.nn as nn
import torchvision.utils
from sympy import false
from torch.optim import Optimizer, Adam, RMSprop
from torch.utils.data import ConcatDataset, DataLoader, sampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
from tqdm import tqdm

import xml.etree.ElementTree as ET
from discriminatorTeachersModels.DiscriminatorTeacher import DiscriminatorTeacher
from experiment.EvaluateLectureGAN import EvaluateLectureGAN
from explanation_tools.ExplanationAlgorithm import ExplanationAlgorithm
from explanation_tools.ExplanationLog import ExplanationLog
from explanation_tools.ExplanationUtils import ExplanationUtils
from generatorStudentsModels.GeneratorNN import GeneratorNN
from generatorStudentsModels.GeneratorStudent import GeneratorStudent
from trainingMethods.discriminator.DiscriminatorTrainingAlgorithm import DiscriminatorTrainingAlgorithm
from trainingMethods.generator.GeneratorTrainingAlgorithm import GeneratorTrainingAlgorithm
from utils.tensor_utils import get_device, normalise_tensor_uint8, save_01Elems_as_Image, resetDirectory, device, \
    createSavingDirectory, createSavingModelPthFile
from workingDatasets.WorkingDataset import WorkingDataset


class LectureGAN:

    def __init__(self, schoolGANId:str, lectureClassId: str, generatorStudent: GeneratorStudent, discriminatorTeacher: DiscriminatorTeacher, workingDataset: WorkingDataset,
                 generatorTrainingAlgo: GeneratorTrainingAlgorithm, discriminatorTrainingAlgo: DiscriminatorTrainingAlgorithm, explanationAlgorithm: ExplanationAlgorithm,
                 experimentParams: Dict[str, ET], explanationParams: Dict[str, ET], evaluationParams: Dict[str, ET],
                 genParamsTrainAlgo: Dict[str, ET], discParamsTrainAlgo: Dict[str, ET]):
        self.__schoolGANId: str = schoolGANId
        self.__lectureClassId: str = lectureClassId
        self.__generatorStudent: GeneratorStudent = generatorStudent
        self.__discriminatorTeacher: DiscriminatorTeacher = discriminatorTeacher
        self.__workingDataset: WorkingDataset = workingDataset
        self.__explanationAlgorithm: ExplanationAlgorithm = explanationAlgorithm
        self.__experimentParams: Dict[str, ET] = experimentParams
        self.__explanationParams: Dict[str, ET] = explanationParams
        self.__evaluationParams: Dict[str, ET] = evaluationParams
        self.__genParamsTrainAlgo: Dict[str, ET] = genParamsTrainAlgo
        self.__discParamsTrainAlgo: Dict[str, ET] = discParamsTrainAlgo
        self.__generatorTrainingAlgo: GeneratorTrainingAlgorithm = generatorTrainingAlgo
        self.__discTrainingAlgo: DiscriminatorTrainingAlgorithm = discriminatorTrainingAlgo
        self.__EPOCHS = int(self.__experimentParams["epochs"].text)
        self.__BATCH_SIZE = int(self.__experimentParams["batchSize"].text)
        self.__G_LEARNING_RATE = float(self.__experimentParams["glearningRate"].text)
        self.__D_LEARNING_RATE = float(self.__experimentParams["dlearningRate"].text)
        self.__DISC_ITERATIONS = int(self.__experimentParams["discIterations"].text) if "discIterations" in self.__experimentParams.keys() else 1
        self.__IMAGE_SIZE = int(self.__experimentParams["imageSize"].text)
        self.__NUM_CHANNELS = int(self.__experimentParams["numChannels"].text)
        self.__PERCENTAGE = float(self.__experimentParams["trainingPercentage"].text)
        self.__USE_EXPLAINABILITY = eval(self.__experimentParams["useExplainability"].text)
        self.__FAKE_LABEL = float(self.__experimentParams["fakeLabel"].text)
        self.__REAL_LABEL = float(self.__experimentParams["realLabel"].text)
        self.__ENABLE_EVAL = eval(self.__evaluationParams["enableEvaluation"].text)
        self.__NUM_EVAL_SAMPLES = int(self.__evaluationParams["numEvaluationSamples"].text)
        self.__LOSS: nn.Module = None
        self.__GEN_OPTIMIZER = None
        self.__DISC_OPTIMIZER = None

        self.__setupLoss(self.__experimentParams["loss"])
        self.__setupGenOptimizer(self.__experimentParams["genOptimizer"])
        self.__setupDiscOptimizer(self.__experimentParams["discOptimizer"])

        self.__writer_real = SummaryWriter(os.path.join('logs', self.__schoolGANId, self.__lectureClassId, "real"))
        self.__writer_fake = SummaryWriter(os.path.join('logs', self.__schoolGANId, self.__lectureClassId, "fake"))
        self.__writer_loss = SummaryWriter(os.path.join('logs', self.__schoolGANId, self.__lectureClassId, "loss"))

        # overwrite the pth files to empty ones
        self.__setupDirectoriesForSavingModels()
        # Reset the results directories.
        # TO DO: If the running model is just loaded, make a mechanism to skip this step to prevent losing the metrics
        #self.__setupDirectoriesForSavingResults()

    def __setupLoss(self, lossTypeXML: ET) -> None:
        lossType = lossTypeXML.text
        if lossType == 'BCELoss':
            self.__LOSS = nn.BCELoss().to(device=get_device())
        elif lossType == 'BCEWithLogitsLoss':
            self.__LOSS = nn.BCEWithLogitsLoss().to(device=get_device())
        elif lossType == 'MSELoss':
            self.__LOSS = nn.MSELoss().to(device=get_device())

    def __setupDiscOptimizer(self, optimizerTypeXML: ET) -> None:
        optimizerType = optimizerTypeXML.text
        if optimizerType == 'Adam':
            self.__DISC_OPTIMIZER = Adam(self.__discriminatorTeacher.parameters(), self.__D_LEARNING_RATE, betas=(0.5, 0.99))
        elif optimizerType == 'RMSprop':
            self.__DISC_OPTIMIZER = RMSprop(self.__discriminatorTeacher.parameters(), self.__D_LEARNING_RATE)

    def __setupGenOptimizer(self, optimizerTypeXML: ET) -> None:
        optimizerType = optimizerTypeXML.text
        if optimizerType == 'Adam':
            self.__GEN_OPTIMIZER = Adam(self.__generatorStudent.parameters(), self.__G_LEARNING_RATE, betas=(0.5, 0.99))
        elif optimizerType == 'RMSprop':
            self.__GEN_OPTIMIZER = RMSprop(self.__generatorStudent.parameters(), self.__G_LEARNING_RATE)

    def __setupExplanationEnvLecture(self):
        ExplanationUtils.setWeightExplanationGrad(float(self.__explanationParams["weightExplanationGradient"].text))
        ExplanationUtils.setGeneratorStudent(self.__generatorStudent)

        self.__explanationLog: ExplanationLog = ExplanationLog(self.__schoolGANId, self.__lectureClassId)
        self.__explanationAlgorithm.setExplanationLog(self.__explanationLog)

    def __setupDirectoriesForSavingModels(self):
        # create the root saving directory where all pth file for a specif model are saved
        self.__saving_best_model_path = createSavingDirectory(self.__schoolGANId, self.__lectureClassId)
        # self.__saving_best_model_path_file = os.path.join(self.__saving_best_model_path, 'best_' + self.__lectureClassId + '.pth')
        # with open(self.__saving_best_model_path_file, "w") as f:
        #     pass

        # self.__saving_last_model_path_file = os.path.join(self.__saving_best_model_path, 'last_' + self.__lectureClassId + '.pth')
        # with open(self.__saving_last_model_path_file, "w") as f:
        #     pass

    def getSavedLastModelPath(self) ->str:
        return self.__saving_last_model_path_file

    def getSavedBestModelPath(self) -> str:
        return self.__saving_best_model_path_file

    def __setupDirectoriesForSavingResults(self):
        self.__results_path = os.path.join('results', self.__schoolGANId, self.__lectureClassId)
        resetDirectory(self.__results_path)

        self.__results_path_saving_images_counterfeit = os.path.join(self.__results_path, 'counterfeitLast')
        resetDirectory(self.__results_path_saving_images_counterfeit)

        self.__results_path_saving_images_best = os.path.join(self.__results_path, 'counterfeitBest')
        resetDirectory(self.__results_path_saving_images_best)

        self.__results_path_saving_images_real = os.path.join(self.__results_path, 'realImages')
        resetDirectory(self.__results_path_saving_images_real)

    def getLectureClassId(self) -> str:
        return self.__lectureClassId

    def getBatchSize(self) -> int:
        return self.__BATCH_SIZE

    def setBatchSize(self, batchSize: int) -> None:
        self.__batchSize = batchSize

    def getGeneratorStudent(self) -> GeneratorStudent:
        return self.__generatorStudent

    def getWorkingDataset(self) -> WorkingDataset:
        return self.__workingDataset

    def __evaluateLectureClass(self) -> None:

        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< EvaluateLectureClass")
        evaluateLectureGAN = EvaluateLectureGAN()

        realSamples = self.__workingDataset.generateRealSamples(self.__NUM_EVAL_SAMPLES)
        save_01Elems_as_Image(realSamples, os.path.join(self.__results_path, 'realImages'))
        self.__generatorStudent.eval()
        counterfeitSamples = self.__generatorStudent.generateCounterfeitSamples(self.__NUM_EVAL_SAMPLES, normalizing="01")
        save_01Elems_as_Image(counterfeitSamples, self.__results_path_saving_images_counterfeit)
        self.__generatorStudent.train()

        #fidLastEpoch = evaluateLectureGAN.evaluateFIDScoresDataProvided(counterfeitSamples, realSamples)
        #isLastEpoch = evaluateLectureGAN.evaluateISScoreDataProvided(normalise_tensor_uint8(counterfeitSamples))

        # deepcopy = preserving the original state of the original generator
        # BEST MODEL
        # best_epoch_generator = deepcopy(self.__generatorStudent).to(device=get_device())
        # best_epoch_generator.load_state_dict(torch.load(self.__saving_best_model_path_file, map_location=get_device()))
        # best_epoch_generator.eval()
        # counterfeitSamplesBestEpoch = best_epoch_generator.generateCounterfeitSamples(self.__NUM_EVAL_SAMPLES, normalizing="01")
        # save_01Elems_as_Image(counterfeitSamplesBestEpoch, self.__results_path_saving_images_best)

        #fidBestEpoch = evaluateLectureGAN.evaluateFIDScoresDataProvided(counterfeitSamplesBestEpoch, realSamples)
        #isBestEpoch = evaluateLectureGAN.evaluateISScoreDataProvided(normalise_tensor_uint8(counterfeitSamplesBestEpoch))

        with open(os.path.join(self.__results_path, 'scores.txt'), "w") as f:
            pass
            #f.write(f"FID last epoch: {fidLastEpoch}\n")
            #f.write(f"IS last epoch: {isLastEpoch}\n")
            #f.write(f"FID best epoch: {fidBestEpoch}\n")
            #f.write(f"IS best epoch: {isBestEpoch}\n")

        # del best_epoch_generator
        del counterfeitSamples
        # del counterfeitSamplesBestEpoch

    def runExperiment(self):
        # due to everything in ExplanationUtils is static, we cannot make the call of the method in the constructor because we are initialise
        # every lecture class from the beginning. So, ExplanationUtils state is going to be overrode by each new instance of LectureGAN
        # Due to we run each LectureGAN sequentially, we initialise the explanation env when we run the training process. This solution had to be considered
        # when we had that code version where we set up all the lectures classes, put them into a list and iterate through them to start training
        if self.__USE_EXPLAINABILITY:
            self.__setupExplanationEnvLecture()

        # EPOCHS = 4 -> explainability_kick_in = 2; EPOCHS = 5 -> explainability_kick_in = 3
        explanationSwitch = (self.__EPOCHS + 1) // 2 if self.__EPOCHS % 2 == 1 else self.__EPOCHS // 2

        upperLimitSamples = int(self.__PERCENTAGE * len(self.__workingDataset))
        indices = [i for i in range(upperLimitSamples)]
        loader = DataLoader(self.__workingDataset, batch_size=self.__BATCH_SIZE, sampler=sampler.SubsetRandomSampler(indices))

        # we have already moved the generator, discriminator, and the loss to the GPU
        self.__generatorStudent.train()
        self.__discriminatorTeacher.train()

        trained_data_explanation = None
        if self.__USE_EXPLAINABILITY:
            trained_data_explanation = next(iter(loader))[0].to(device=get_device())

        local_explainable = False
        logging_step = 1
        SAVE_REAL_IMAGES = True  # set immediately to false once the real images were saved once
        fixed_noise = self.__generatorStudent.generateNoise(batch_size=self.__BATCH_SIZE).to(device=get_device())

        # Start training for Epochs [1, self.__EPOCHS + 1]
        for epoch in range(1, self.__EPOCHS + 1):

            if self.__USE_EXPLAINABILITY and (epoch - 1) == explanationSwitch:
                # TO DO: register hook for shaping the gradient descent in a way that focuses generator training
                # on input features the discriminator recognizes as important

                '''
                register_backward_hook vs register_full_backward_hook
                register_backward_hook only alters the gradients corresponding to the out layer of the generator
                register_full_backward_hook alters the gradients from all of the layers of the generator
                '''
                self.__generatorStudent.getOutLayer().register_backward_hook(ExplanationUtils.explanation_hook)
                local_explainable = True

            loop = tqdm(loader, leave=True)
            if self.__USE_EXPLAINABILITY:
                self.__explanationLog.setEpoch(epoch)
            mini_generative_loss = float('inf')
            for batch_idx, (real_batch, _) in enumerate(loop):
                # the last batch can be smaller than the experiment config
                BATCH_SIZE = real_batch.size(0)

                real_batch = real_batch.to(device=get_device())

                for _ in range(self.__DISC_ITERATIONS):
                    # generate fake data
                    noise = self.__generatorStudent.generateNoise(batch_size=BATCH_SIZE).to(device=get_device())
                    # DETACH() is required such that to not compute the generator's gradients. If we don't detach, when we backpropagate the error for the disc
                    # the explainability hook (in the second half of training) will be invoked because the error comprises an error from the generator side as well
                    fake_data = self.__generatorStudent(noise).detach().to(device=get_device())

                    # Train Discriminator
                    d_error, d_pred_real, d_pred_fake = self.__discTrainingAlgo.trainDiscriminator(self.__discriminatorTeacher, real_batch, fake_data,
                                                                                                   self.__LOSS, self.__DISC_OPTIMIZER, self.__REAL_LABEL, self.__FAKE_LABEL,
                                                                                                   self.__discParamsTrainAlgo)

                # generate fake data for generator
                noise = self.__generatorStudent.generateNoise(batch_size=BATCH_SIZE).to(device=get_device())
                fake_data = self.__generatorStudent(noise).to(device=get_device())

                # train G
                g_error = self.__generatorTrainingAlgo.trainGenerator(self.__discriminatorTeacher, self.__generatorStudent, fake_data, local_explainable,
                                                                      trained_data_explanation, self.__explanationAlgorithm,
                                                                      self.__LOSS, self.__GEN_OPTIMIZER, self.__REAL_LABEL,
                                                                      self.__genParamsTrainAlgo)

                # Write To Tensorboard
                if len(loader) == 1 or (batch_idx > 0 and (batch_idx % 2 == 0 or batch_idx == len(loader) - 1)):
                    self.__generatorStudent.eval()
                    with torch.no_grad():
                        fake_inference_images = self.__generatorStudent(fixed_noise[:self.__BATCH_SIZE]).to(device=get_device())
                        fake_inference_images = fake_inference_images.reshape(-1, self.__NUM_CHANNELS, self.__IMAGE_SIZE, self.__IMAGE_SIZE)
                        nr_images_to_save = min(32, self.__BATCH_SIZE)

                        if SAVE_REAL_IMAGES:
                            SAVE_REAL_IMAGES = False
                            img_grid_real = torchvision.utils.make_grid(real_batch[:nr_images_to_save], normalize=True)
                            self.__writer_real.add_image(f"id_Img_Real", img_grid_real, global_step=logging_step)

                        img_grid_fake = torchvision.utils.make_grid(fake_inference_images[:nr_images_to_save], normalize=True)
                        self.__writer_fake.add_image(f"id_Img_Fake", img_grid_fake, global_step=logging_step)

                        del fake_inference_images

                    # Plot losses
                    self.__writer_loss.add_scalar(f"id_Loss_Generator", g_error, global_step=logging_step)
                    self.__writer_loss.add_scalar(f"id_Loss_Discriminator", d_error, global_step=logging_step)

                    self.__generatorStudent.train()
                    logging_step += 1

                # saving generative model
                if g_error <= mini_generative_loss: # not < because we want a ore experienced generator
                    mini_generative_loss = g_error
                    self.__saveGenerativeModel()

                loop.set_postfix(Epoch=epoch, g_error=g_error.item(), d_error=d_error.item())

            if epoch >= 50: # To get rid of the beginning models that generate most of the noisy samples
                self.__saveGenerativeModelLastEpoch(epoch)

        # if self.__ENABLE_EVAL:
        #     self.__evaluateLectureClass()

    def __saveGenerativeModel(self):
        pass
        #torch.save(self.__generatorStudent.state_dict(), self.__saving_best_model_path_file)

    def __saveGenerativeModelLastEpoch(self, epoch: int):
        saving_last_model_path_file = createSavingModelPthFile(self.__saving_best_model_path, epoch)
        torch.save(self.__generatorStudent.state_dict(), saving_last_model_path_file)

    def freeUpLectureClass(self):
        del self.__generatorStudent
        del self.__discriminatorTeacher
        del self.__workingDataset
        del self.__experimentParams
        del self.__explanationParams
        del self.__evaluationParams
        del self.__genParamsTrainAlgo
        del self.__discParamsTrainAlgo


