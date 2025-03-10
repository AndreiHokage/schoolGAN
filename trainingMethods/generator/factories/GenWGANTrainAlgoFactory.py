from trainingMethods.generator.GenWGANTrainAlgo import GenWGANTrainAlgo
from trainingMethods.generator.GeneratorTrainingAlgorithm import GeneratorTrainingAlgorithm
from trainingMethods.generator.factories.GenTrainAlgoAbstractFactory import GenTrainAlgoAbstractFactory
from xmlComponents.GenTrainAlgoXML import GenTrainAlgoXML


class GenWGANTrainAlgoFactory(GenTrainAlgoAbstractFactory):

    def _instantiateGenTrainingAlgo(self, genTrainAlgoXML: GenTrainAlgoXML) -> GeneratorTrainingAlgorithm:
        generatorTrainingAlgorithm = GenWGANTrainAlgo()
        return generatorTrainingAlgorithm

