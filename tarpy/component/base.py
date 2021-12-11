from dataclasses import dataclass
from typing import List, Type
from ..base import Savable
from functools import partial

"""
Note:
If component involves randomness, random seed should be set in `begin` method.
"""

class Component(Savable):

    def __init__(self):
        super().__init__()
    
    def begin(self, *args, **kwargs):
        pass

    def reset(self, *args, **kwargs):
        pass

    @property
    def hasStoppingRule(self):
        return hasattr(self, 'checkStopping')
    
    @property
    def hasRanker(self):
        return hasattr(self, 'trainRanker') and hasattr(self, 'scoreDocuments')

    @property
    def hasLabeler(self):
        return hasattr(self, 'labelDocs')
    
    @property
    def hasSampler(self):
        return hasattr(self, 'sampleDocs')
    
    @property
    def __name__(self):
        if hasattr(self, 'name'):
            return self.name
        return self.__class__.__name__


class FunctionComponent(Component):

    def __init__(self, method, func, **kwargs):
        super().__init__()
        setattr(self, method, partial(func, **kwargs))
        self.name = func.__name__
    

def asComponent(method):
    def wrapper(func):
        def factory(**kwargs):
            return FunctionComponent(method, func, **kwargs)
        factory.__name__ = func.__name__
        return factory
    return wrapper

@dataclass(frozen=True)
class CombinedComponent(Component):
    _comps: List[Component]

    def begin(self, *args, **kwargs):
        for comp in self._comps:
            comp.begin(*args, **kwargs)
    
    def reset(self, *args, **kwargs):
        for comp in self._comps:
            comp.reset(*args, **kwargs)
    
    def __getattr__(self, attr):
        if attr.startswith('_'): 
            raise AttributeError
        for comp in self._comps:
            if hasattr(comp, attr):
                return getattr(comp, attr)
        raise AttributeError
    

def combine(*comps):
    return lambda **kwargs: CombinedComponent(comps)