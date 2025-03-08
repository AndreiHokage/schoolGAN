from trainingMethods.discriminator.DiscClassicTrainAlgo import DiscClassicTrainAlgo
from trainingMethods.discriminator.DiscriminatorTrainingAlgorithm import DiscriminatorTrainingAlgorithm
from trainingMethods.discriminator.factories.DiscTrainAlgoAbstractFactory import DiscTrainAlgoAbstractFactory
from xmlComponents.DiscTrainAlgoXML import DiscTrainAlgoXML


class DiscClassicTrainAlgoFactory(DiscTrainAlgoAbstractFactory):

    def _instantiateDiscTrainingAlgo(self, discTrainAlgoXML: DiscTrainAlgoXML) -> DiscriminatorTrainingAlgorithm:
        discriminatorTrainingAlgorithm = DiscClassicTrainAlgo()
        return discriminatorTrainingAlgorithm

