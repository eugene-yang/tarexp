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