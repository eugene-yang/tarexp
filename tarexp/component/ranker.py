"""``TARexp`` supports classification models implemented in Scikit-learn through 
:py:class:`tarexp.component.ranker.SklearnRanker` wrapper. However, any supervised 
learning model that can produce a score for each document in the collection can be 
integrated into ``TARexp``. 

.. caution::
    For rankers that require allocating a lot of memory or carries states 
    (e.g. neural models and SGDClassifier in sklearn that supports partial_fit), 
    it would be ideal to put the actual model initialization into ``.begin`` method 
    and properly dispose the model instance in ``.reset`` method.

"""

import warnings
from copy import deepcopy

import numpy as np
from tarexp.component.base import Component

class Ranker(Component):
    
    def __init__(self, **kwargs):
        super().__init__()
    
    def trainRanker(self, X, y, *args, **kwargs):
        raise NotImplementedError
    
    def scoreDocuments(self, X, *args, **kwargs):
        raise NotImplementedError

class SklearnRanker(Ranker):

    def __init__(self, module, **kwargs):
        super().__init__()
        assert hasattr(module, 'fit')
        if not hasattr(module, 'predict_proba'):
            warnings.warn("Model that supports predicting probabilities is preferable. "
                          "Will invoke `decision_function` instead.")
            assert hasattr(module, 'decision_function')
        self.sk_module = module 
        self._model_kwargs = kwargs
        self.reset()
    
    def reset(self):
        # make sure model starts fresh so other component can safely take
        # advantage of the state of the model as input
        self.model = self.sk_module(**self._model_kwargs)
    
    def trainRanker(self, X, y, **kwargs):
        assert X.shape[0] == len(y)
        if np.unique(y).size == 1:
            # fix for sklearn models that does not support one-class classification
            X, y = addDummyNeg(X, y)
        assert np.unique(y).size == 2
        return self.model.fit(X, y)
    
    def scoreDocuments(self, X, **kwargs):
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            return self.model.decision_function(X)[:, 1]


def addDummyNeg(X, y):
    import scipy.sparse as sp

    assert X.shape[0] == len(y)
    assert np.unique(y).size == 1
    if isinstance(X, np.ndarray):
        X = np.vstack([X, np.zeros((1, X.shape[1]))])
    elif sp.issparse(X):
        X = sp.vstack([X, sp.csr_matrix((1, X.shape[1]))])
    return X, list(y) + [False]

