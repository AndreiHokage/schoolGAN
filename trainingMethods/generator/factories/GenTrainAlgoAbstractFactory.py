from trainingMethods.generator.GeneratorTrainingAlgorithm import GeneratorTrainingAlgorithm
from xmlComponents.GenTrainAlgoXML import GenTrainAlgoXML


class GenTrainAlgoAbstractFactory:

    def createGenTrainingAlgo(self, genTrainAlgoXML: GenTrainAlgoXML) -> GeneratorTrainingAlgorithm:
        generatorTrainingAlgorithm = self._instantiateGenTrainingAlgo(genTrainAlgoXML)
        return generatorTrainingAlgorithm

    def _instantiateGenTrainingAlgo(self, genTrainAlgoXML: GenTrainAlgoXML) -> GeneratorTrainingAlgorithm:
        pass

