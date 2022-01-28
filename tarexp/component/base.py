from warnings import warn
from pathlib import Path
from typing import List
from functools import partial

from tarexp.base import Savable, easy_repr
from tarexp.util import saveObj, readObj


"""
Note:
If component involves randomness, random seed should be set in `begin` method.
"""
@easy_repr
class Component(Savable):
    
    def begin(self, *args, **kwargs):
        pass

    def reset(self, *args, **kwargs):
        pass

    def checkRole(self, role):
        return hasattr(self, role)

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


class CombinedComponent(Component):

    def __init__(self, comps: List[Component], strict: bool= True):
        self._comps: List[Component] = comps
        existing_methods = set()
        for comp in self._comps:
            methods = set([ 
                a for a in dir(comp) 
                if not a.startswith('_') and a not in ['begin', 'reset', 'save', 'load'] and callable(a)
            ])
            unique_methods = (methods - existing_methods)
            if len(unique_methods) != len(methods):
                mesg = f"Cannot resolve methods `{methods - unique_methods}` when combining."
                if strict:
                    raise AttributeError(mesg)
                else:
                    warn(mesg + " Invoking these methods will result in arbitrary resolution.")
            existing_methods &= methods

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
                val = getattr(comp, attr)
                if callable(val):
                    return partial(val, __combined_component__=self)
                return val
        raise AttributeError
    
    @property
    def __name__(self):
        return " + ".join([ repr(comp) for comp in self._comps ])

    def __repr__(self):
        return self.__class__.__name__ + f"[{self.__name__}]"
    
    def save(self, output_file: Path):
        if output_file.is_dir():
            output_file = output_file / "component.meta.pgz"
        prefix = output_file.stem
        output_dir = output_file.parent

        references = []
        for comp in self._comps:
            fn = output_dir / f"{prefix}.{comp.__name__}"
            references.append((comp.__class__, fn.name))
            comp.save(fn)

        saveObj(references, output_file)
        return output_file
    
    @classmethod
    def load(cls, path: Path):
        if path.is_dir():
            path = path / "component.meta.pgz"
        
        return cls([
            subcls.load(path.parent / fn)
            for subcls, fn in readObj(path, list)
        ])
        

def combine(*comps):
    return lambda **kwargs: CombinedComponent(comps)