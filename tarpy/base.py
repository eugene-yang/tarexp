from collections import defaultdict
from os import read
from .util import saveObj, readObj

class Eventable:

    def __init__(self):
        self._callbacks = defaultdict(list)
        self.children = None

    def fire(self, event: str, *args, **kwargs):
        if self.children is not None:
            for c in self.children:
                c.fire(event, *args, **kwargs)
        if event in self._callback:
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
    def save(self, fn):
        saveObj(self, fn)
    
    @classmethod
    def load(cls, fn):
        return readObj(fn, cls)
    

