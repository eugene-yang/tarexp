"""A :py:class:`tarexp.component.labeler.Labeler` instance simulates human review. 
For most TAR experiments, we assume we simply look up the gold label of each document using 
:py:class:`tarexp.component.labeler.PerfectLabeler`. 
Random errors can be introduced using :py:class:`tarexp.component.labeler.SuccessProbLabeler`. 
"""

from tarexp.component.base import Component
from tarexp.dataset import Dataset
import numpy as np

class Labeler(Component):
    def __init__(self, **kwargs):
        super().__init__()
    
    def labelDocs(self, doc_ids, *args, **kwargs):
        raise NotImplementedError
    
class PerfectLabeler(Labeler):
    def __init__(self, **kwargs):
        super().__init__()
        self.reset()
    
    def begin(self, dataset: Dataset, **kwargs):
        self._Y = np.asanyarray(dataset.labels)
    
    def reset(self):
        self._Y = None
    
    def labelDocs(self, doc_ids, **kwargs):
        return list(self._Y[doc_ids])

class SuccessProbLabeler(Labeler):
    def __init__(self, success_prob=0.5, **kwargs):
        super().__init__()
        self._success_prob = success_prob
        self.reset()
    
    def begin(self, dataset: Dataset, **kwargs):
        self._Y = np.asanyarray(dataset.labels)
    
    def reset(self):
        self._Y = None
    
    def labelDocs(self, doc_ids, random: np.random.RandomState, **kwargs):
        labs = self._Y[doc_ids]
        flip = random.uniform(size=len(labs)) > self._success_prob
        return list(np.bitwise_xor(labs, flip))