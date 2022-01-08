from typing import Iterable

import pickle
import gzip
from xxhash import xxh128_hexdigest

def stable_hash(obj):
    # it is actually pretty important for save/load sanity check
    return xxh128_hexdigest(obj)

def getOneDimScores(scores):
    if len(scores.shape) == 1:
        assert scores.min() >= 0 and scores.max() <= 1
        return scores
    elif len(scores.shape) == 2:
        return scores[:, 1]
    raise ValueError("Scores should be either probabilities of positive ",
                     "or probabilities of all classes. ")

def saveObj(obj, fn):
    with gzip.open(fn, 'wb') as fw:
        pickle.dump(obj, fw)

def readObj(fn, cls=None):
    with gzip.open(fn, 'rb') as fr:
        obj =  pickle.load(fr)
    if cls is not None:
        assert isinstance(obj, cls)
    return obj

class iter_with_length(Iterable):
    def __init__(self, it: Iterable, length: int = None):
        if hasattr(it, '__len__'):
            self._len = len(it)
        else: 
            assert length is not None
            self._len = length
        self._iter = it
    
    def __iter__(self):
        return iter(self._iter)

    def __len__(self):
        return self._len

