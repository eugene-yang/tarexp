from argparse import Namespace
from collections import defaultdict
from pathlib import Path

from tarexp.util import saveObj, readObj, stable_hash

class Eventable:

    def __init__(self):
        self._callbacks = defaultdict(list)
        self.children = None

    def fire(self, event: str, *args, **kwargs):
        if self.children is not None:
            for c in self.children:
                c.fire(event, *args, **kwargs)
        if event in self._callbacks:
            for f in self._callbacks[event]:
                f(*args, **kwargs)
    
    def on(self, event: str, f: callable):
        if event not in self._callbacks:
            self._callbacks[event] = [f]
        else:
            self._callbacks.append(f)
    
    def updateCallbacks(self, callbacks):
        for event, funcs in callbacks.items():
            self._callbacks[event] += funcs if isinstance(funcs, list) else [funcs]

class Savable:
    # By default we just pickle everything
    # but each module should implement its own save/load methods
    # for space efficiency and better usability.
    # This is just a fall back approach. 
    def save(self, path: Path, overwrite=False) -> Path:
        if path.is_dir():
            path = path / f"{self.__class__.__name__.lower()}.pgz"
        if not overwrite and path.exists():
            raise FileExistsError
        saveObj(self, path)
        return path

    @classmethod
    def load(cls, path: Path):
        # allow both a directory then infer the name and directly the file name
        if path.is_dir():
            path = path / f"{cls.__name__.lower()}.pgz"
        return readObj(path, cls)

class space(Namespace):
    def __init__(self, data={}, hashable_only=True):
        for k, v in data.items():
            self.__setattr__(k, v)

    def keys(self):
        return tuple(sorted(self.__dict__.keys()))
    
    def __iter__(self):
        return iter(self.keys())
    
    def items(self):
        return iter((k, self.__dict__[k]) for k in self.keys())
    
    def values(self):
        return tuple(self.__dict__[k] for k in self.keys())

    def __repr__(self):
        if len(self) == 0:
            return ''
        return "[" + ", ".join([ f"{k}={v}" for k, v in self.items() ]) + "]"
    
    def __hash__(self):
        return stable_hash(repr(self))
    
    def __len__(self):
        return len(self.__dict__)
    
    def __getitem__(self, name):
        return self.__dict__[name]


def _make_repr(module):
    cls_name = module.__name__ if hasattr(module, '__name__') else str(module.__class__)
    if hasattr(module, 'config') and \
        isinstance(getattr(module, 'config'), space):
        config = getattr(module, 'config')
    else:
        # relying on variable naming rules
        config = space({
            k: getattr(module, k)
            for k in dir(module) 
            if hasattr(k, '__hash__') and 
               not callable(getattr(module, k)) 
               and k != 'self' 
               and not k.startswith('_')
               and not k.startswith('has')
        })

    return cls_name + repr(config)

def easy_repr(cls):
    setattr(cls, '__repr__', _make_repr)
    return cls