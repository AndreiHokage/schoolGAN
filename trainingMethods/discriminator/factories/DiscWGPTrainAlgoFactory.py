from trainingMethods.discriminator.DiscWGPTrainAlgo import DiscWGPTrainAlgo
from trainingMethods.discriminator.DiscriminatorTrainingAlgorithm import DiscriminatorTrainingAlgorithm
from trainingMethods.discriminator.factories.DiscTrainAlgoAbstractFactory import DiscTrainAlgoAbstractFactory
from xmlComponents.DiscTrainAlgoXML import DiscTrainAlgoXML


class DiscWGPTrainAlgoFactory(DiscTrainAlgoAbstractFactory):

    def _instantiateDiscTrainingAlgo(self, discTrainAlgoXML: DiscTrainAlgoXML) -> DiscriminatorTrainingAlgorithm:
        discriminatorTrainingAlgorithm = DiscWGPTrainAlgo()
        return discriminatorTrainingAlgorithm

