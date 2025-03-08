from trainingMethods.generator.GenClassicTrainAlgo import GenClassicTrainAlgo
from trainingMethods.generator.GeneratorTrainingAlgorithm import GeneratorTrainingAlgorithm
from trainingMethods.generator.factories.GenTrainAlgoAbstractFactory import GenTrainAlgoAbstractFactory
from xmlComponents.GenTrainAlgoXML import GenTrainAlgoXML


class GenClassicTrainAlgoFactory(GenTrainAlgoAbstractFactory):

    def _instantiateGenTrainingAlgo(self, genTrainAlgoXML: GenTrainAlgoXML) -> GeneratorTrainingAlgorithm:
        generatorTrainingAlgorithm = GenClassicTrainAlgo()
        return generatorTrainingAlgorithm

