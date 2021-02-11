from typing import Dict, List, Union, Optional, Callable
import os
from glob import glob
from typing import Dict, List, Union, Optional, Callable
from collections import namedtuple, OrderedDict


class BaseProvider():

    KIND: str
    NAME: str
    SUFFIXES: List[str]

    def __init__(self, path: Optional[str] = None):
        self.path = path

    def info(self):
        return f"{self.NAME} loads {self.KIND} from files ending with {self.SUFFIXES}."

    @classmethod
    def match(cls, filename):
        if isinstance(filename, str):
            return any([filename.endswith(s) for s in cls.SUFFIXES])
        else:
            return False

    @classmethod
    def get_loader(cls, filename: str):
        if cls.match(filename):
            loader = cls(filename)
        else:
            loader = None
        return loader

    def load(self, filename: Optional[str] = None, **kwargs):
        raise NotImplementedError


# TODO set precedence - maybe with provider.PRIORITY and sort providers accordingly?
providers = []
kinds = OrderedDict()
def register_provider(func):
    providers.append(func)
    if func.KIND not in kinds:
        kinds[func.KIND] = []
    kinds[func.KIND].extend(func.SUFFIXES)



def get_loader(kind: str, basename: str, stop_after_match: bool = True):
    if kind not in kinds:
        raise ValueError(f'Unknow kind {kind} - must be in {kinds}.')

    loaders = []
    for suffix in kinds[kind]:
        paths = glob(basename + suffix)
        for path in paths:
            for provider in providers:
                loader = provider.get_loader(path)
                if loader is not None and loader.KIND == kind:
                    if stop_after_match:
                        return loader
                    else:
                        loaders.append(loader)
    return loaders

from . import (tracks,
               annotations,
               annotations_manual,
               poses,
               audio)