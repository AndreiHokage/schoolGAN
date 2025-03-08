from trainingMethods.discriminator.DiscriminatorTrainingAlgorithm import DiscriminatorTrainingAlgorithm
from xmlComponents.DiscTrainAlgoXML import DiscTrainAlgoXML


class DiscTrainAlgoAbstractFactory:

    def createDiscTrainingAlgo(self, discTrainAlgoXML: DiscTrainAlgoXML) -> DiscriminatorTrainingAlgorithm:
        discriminatorTrainingAlgorithm = self._instantiateDiscTrainingAlgo(discTrainAlgoXML)
        return discriminatorTrainingAlgorithm

    def _instantiateDiscTrainingAlgo(self, discTrainAlgoXML: DiscTrainAlgoXML) -> DiscriminatorTrainingAlgorithm:
        pass


