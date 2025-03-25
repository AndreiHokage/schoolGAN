import xml.etree.ElementTree as ET

from xmlComponents.GeneratorStudentXML import GeneratorStudentXML


class GeneratorTeamXML(GeneratorStudentXML):

    def __init__(self, xmlRoot: ET):
        super().__init__(xmlRoot)

