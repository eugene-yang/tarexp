"""The ``base`` module contains classes and functions that are inherited by 
other ``TARexp`` classes.  
"""

from __future__ import annotations
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

from tarexp.util import saveObj, readObj, stable_hash

class Eventable:
    """
    .. deprecated:: 0.1.3
        The Eventable class is plan to be removed in the future. 
        It is not currently used by any classes.
        The event dispatcher of the :py:class:`tarexp.experiments.Experiment` is implemented
        in itself.
    """

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
    """Default Savable Class Mixin
    
    Classes inheriting this class by default save itself has a gzipped pickle file.
    The files can be read by :py:class:`tarexp.util.readObj`. 

    However, this is designed to be a catch-all method. 
    Each class potentially would implement its own set of ``save`` and ``load``
    methods, such as :py:meth:`tarexp.workflow.Workflow.save` and :py:meth:`tarexp.workflow.Workflow.load`. 
    """

    def save(self, path: Path | str, overwrite=False) -> Path:
        """Saving the object

        Parameters
        ----------
        path
            Path to the output directory or filename. 
            If a directory is provided, the filename is set to be the name of the class 
            with extention ``.pgz``. 
        overwrite
            Whether to overwrite an existing file. 
        
        Returns
        -------
            The path to the saved file. 
        """
        path = Path(path)
        if path.is_dir():
            path = path / f"{self.__class__.__name__.lower()}.pgz"
        if not overwrite and path.exists():
            raise FileExistsError
        saveObj(self, path)
        return path

    @classmethod
    def load(cls, path: Path):
        """Class method for loading the saved file. 

        The method should be used as ``{Class}.load(filename)`` which loads the 
        saved content as a ``{Class}`` instance with type checking.

        Parameters
        ----------
        path
            Path to the output directory or filename. 
            If a directory is provided, it looks for a name of the class 
            with extention ``.pgz``. 
        
        Returns
        -------
            An instance of the evoked class. 
        """
        if path.is_dir():
            path = path / f"{cls.__name__.lower()}.pgz"
        return readObj(path, cls)

class space(Namespace):
    """Hashable Namespace for Parameters.

    It is an extension of argparse.Namespace that supports better ``__repr__`` method 
    and being hashable with a stable hash function based on the content. 

    This class support direct access to the key-value content by attributes and items. 
    Common dictionary interfaces such as ``.keys()`` and ``.items()`` are also implemented. 

    .. caution:: The content should all be hashable or at least the representation (``.__repr__()``) should 
        uniquely identify the object. It is being used to distinguish experiment runs in :py:class:`tarexp.experiments.Experiment` 
        by comparing the identifier of the :py:class:`tarexp.component.Component`.


    """

    def __init__(self, data={}):
        """The class can be initilized by an optional dictionary.
        The content of the dictionary would be recorded as the attributes of the instance. 
        """
        for k, v in data.items():
            self.__setattr__(k, v)

    def keys(self):
        """
        Returns
        -------
        Tuple[Any]
            Sorted tuple of the existing keys in the instance. 
            Similar to the built-in ``.keys()`` method for Python dictionaries. 
        """
        return tuple(sorted(self.__dict__.keys()))
    
    def __iter__(self):
        """
        Returns
        -------
        iterator
            Iterator of the existing keys in sorted order. 
            Similar to the built-in ``.__iter__()`` method for Python dictionaries. 
        """
        return iter(self.keys())
    
    def items(self):
        """
        Returns
        -------
        Tuple[Any]
            Iterator of the item tuples in sorted order. 
            Similar to the built-in ``.items()`` method for Python dictionaries. 
        """
        return iter((k, self.__dict__[k]) for k in self.keys())
    
    def values(self):
        """
        Returns
        -------
        Tuple[Any]
            Tuple of values in the content sorted by the key. 
            Similar to the built-in ``.values()`` method for Python dictionaries. 
        """
        return tuple(self.__dict__[k] for k in self.keys())

    def __repr__(self) -> str:
        """
        Returns
        -------
        str
            String representation of the instance. 
        """
        if len(self) == 0:
            return ''
        return "[" + ", ".join([ f"{k}={v}" for k, v in self.items() ]) + "]"
    
    def __hash__(self) -> str:
        """Stable hash function of the instance. 
        
        The hash is only depending on the content in the instance instead of the actual memory location. 

        Returns
        -------
        str
            Stable hash string of the instance. 
        """
        return stable_hash(repr(self))
    
    def __len__(self) -> int:
        """
        Returns
        -------
        int
            The number of existing key-value pairs in the instance. 
            Similar to the built-in ``.__len__()`` method for Python dictionaries. 
        """
        return len(self.__dict__)
    
    def __getitem__(self, key):
        """
        Returns
        -------
        Any
            The content associated with the key. 
        """
        return self.__dict__[key]


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
    """Decorator for class to have a better ``__repr__``

    If attribute ``config`` exists, it is used as the representation (unique identifier of the instance).
    If not, it creates a py:class:`tarexp.util.space` instance for the non-callable attributes 
    for building the representation. Attributes starts with ``_`` are ignored.
    """
    setattr(cls, '__repr__', _make_repr)
    return cls