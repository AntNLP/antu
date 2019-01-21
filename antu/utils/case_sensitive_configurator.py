from configparser import ConfigParser
from overrides import overrides


class CaseSensConfigParser(ConfigParser):

    def __init__(self, defaults=None):
        ConfigParser.__init__(self, defaults=None)

    @overrides
    def optionxform(self, optionstr):
        return optionstr
