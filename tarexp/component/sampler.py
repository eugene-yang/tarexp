"""``TARexp`` provides reference implementations of a variety of TAR-specific algorithms, 
to aid reproducibility and reduce experimenter work. For instance, uncertainty sampling [1]_, 
relevance feedback [2]_, and simple random sampling batch selection algorithms are provided.   

This interface is inspired by |libact|_, an Active Learning library. 

.. |libact| replace:: ``libact``
.. _libact: https://github.com/ntucllab/libact

.. seealso::
    .. [1] David D. Lewis, and William A. Gale. "A sequential algorithm for training text classifiers." 
           *SIGIR 1994*. Springer, London, 1994.
           `<https://arxiv.org/abs/cmp-lg/9407020>`__
    
    .. [2] Rocchio Algorithm for psuedo relevance feedback
           `<https://en.wikipedia.org/wiki/Rocchio_algorithm>`__
"""

import numpy as np
from tarexp.component.base import Component

from tarexp.ledger import Ledger
from tarexp.util import getOneDimScores

def _removeKnownDocs(idx_list, ledger):
    known = np.where(ledger.annotated)[0]
    return idx_list[ ~np.isin(idx_list, known) ]

class Sampler(Component):
    def __init__(self, **kwargs): 
        super().__init__()
    
    def sampleDocs(self, nask: int, ledger: Ledger, scores, *args, **kwargs):
        raise NotImplementedError

    
class UncertaintySampler(Sampler):
    """Implementing least confidence uncertainty sampling.
    """
    
    def sampleDocs(self, nask: int, ledger: Ledger, scores, **kwargs):
        scores = np.asanyarray(scores)
        if len(scores.shape) == 1:
            assert scores.min() >= 0 and scores.max() <= 1
            scores = np.abs(scores - 0.5)
        elif len(scores.shape) == 2:
            scores = np.max(scores, axis=1)
        else:
            raise ValueError("Scores should be either probabilities of positive ",
                             "or probabilities of all classes. ")
        return _removeKnownDocs(np.argsort(scores), ledger)[:nask]
    
    def freeze(self):
        return self

class RelevanceSampler(Sampler):

    def sampleDocs(self, nask: int, ledger: Ledger, scores, **kwargs):
        scores = getOneDimScores(np.asanyarray(scores))
        return _removeKnownDocs(np.argsort(scores)[::-1], ledger)[:nask]


class RandomSampler(Sampler):

    def begin(self, random_seed=None, *args, **kwargs):
        self._random = np.random.RandomState(random_seed)

    def sampleDocs(self, nask: int, ledger: Ledger, scores=None, dist=None, **kwargs):
        if scores is None and dist is None:
            raise AttributeError("Missing either scores or dist.")
        elif scores is not None and dist is not None:
            assert np.shape(scores)[0] == np.shape(dist)[0]
        elif dist is None:
            dist = np.ones(np.shape(scores)[0]) / len(scores)
        # give known documents 0 probability
        dist[ ledger.annotated ] = 0
        dist = dist / dist.sum()
        return self._random.choice(np.shape(dist)[0], nask, p=dist, replace=False)
        

        
        