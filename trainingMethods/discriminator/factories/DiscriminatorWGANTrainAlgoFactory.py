from trainingMethods.discriminator.DiscWGANTrainAlgo import DiscWGANTrainAlgo
from trainingMethods.discriminator.DiscriminatorTrainingAlgorithm import DiscriminatorTrainingAlgorithm
from trainingMethods.discriminator.factories.DiscTrainAlgoAbstractFactory import DiscTrainAlgoAbstractFactory
from xmlComponents.DiscTrainAlgoXML import DiscTrainAlgoXML


class DiscriminatorWGANTrainAlgoFactory(DiscTrainAlgoAbstractFactory):

    def _instantiateDiscTrainingAlgo(self, discTrainAlgoXML: DiscTrainAlgoXML) -> DiscriminatorTrainingAlgorithm:
        discriminatorTrainingAlgorithm = DiscWGANTrainAlgo()
        return discriminatorTrainingAlgorithm
