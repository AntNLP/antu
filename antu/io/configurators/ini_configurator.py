from typing import Dict, List, TypeVar
from antu.utils.case_sensitive_configurator import CaseSensConfigParser
import argparse, os, ast

BaseObj = TypeVar("BaseObj", int, float, str, list, set, dict)
BASEOBJ = {int, float, str, list, set, dict}

def str_to_baseobj(s: str) -> BaseObj:
    """
    Converts a string to the corresponding base python type value.

    Parameters
    ----------
    s : ``str``
        string like "123", "12.3", "[1, 2, 3]" ...

    Returns
    -------
    ret : ``BaseObj``
        "123" -> int(123)
        "12.3" -> float(12.3)
        ...
    """
    try:
        ret = ast.literal_eval(s)
    except BaseException:
        return s
    if (s in globals() or s in locals()) and type(ret) not in BASEOBJ:
        return s
    else: return ret


class IniConfigurator:
    """
    Reads and stores the configuration in the ini Format file.

    Parameters
    ----------
    config_file : ``str``
        Path to the configuration file.
    extra_args : ``Dict[str, str]``, optional (default=``dict()``)
        The configuration of the command line input.
    """
    def __init__(
        self,
        config_file: str,
        extra_args: Dict[str, str]=dict()) -> None:

        config = CaseSensConfigParser()
        config.read(config_file)
        if extra_args:
            extra_args = (dict([(k[2:], v)
                for k, v in zip(extra_args[0::2], extra_args[1::2])]))
        attr_name = set()
        for section in config.sections():
            for k, v in config.items(section):
                if k in extra_args:
                    v = type(v)(extra_args[k])
                    config.set(section, k, v)

                if k in attr_name:
                    raise RuntimeError('Attribute (%s) has already '
                                       'appeared.' % (k))
                else: attr_name.update(k)
                super(IniConfigurator, self).__setattr__(k, str_to_baseobj(v))

        with open(config_file, 'w') as fout:
            config.write(fout)

        print('Loaded config file sucessfully.')
        for section in config.sections():
            for k, v in config.items(section):
                print(k, v)

    def __setattr__(self, name, value):
        raise RuntimeError('Try to set the attribute (%s) of the constant '
                           'class (%s).' % (name, self.__class__.__name__))