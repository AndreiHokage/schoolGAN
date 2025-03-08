import os
import torch

from torch.utils.tensorboard import SummaryWriter

class ExplanationLog:

    def __init__(self, schoolGANId: str, lectureClassId: str):
        self.__schoolGANId: str = schoolGANId
        self.__lectureClassId: str = lectureClassId
        self.__writerXAI: SummaryWriter = SummaryWriter(os.path.join('logs', self.__schoolGANId, self.__lectureClassId, "XAI"))
        self.__epoch: int = 0
        self.__step: int = 0

    def incrementStep(self) -> None:
        self.__step += 1

    def addImage(self, image: torch.Tensor) -> None:
        self.__writerXAI.add_image(f"XAI_epoch_{self.__epoch}", image, global_step=self.__step)
        self.incrementStep()

    def getWriterXAI(self) -> SummaryWriter:
        return self.__writerXAI

    def setEpoch(self, epoch: int) -> None:
        self.__epoch = epoch
        self.__step = 0

