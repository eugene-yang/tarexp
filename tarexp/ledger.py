"""Any aspect of the history of a batch-based workflow can, if necessary, be reproduced from a record of 
which documents were labeled on which training rounds (including any initial seed round). 
The :py:class:`tarexp.ledger.Ledger` instance records this state in memory, and 
writes it to disk at user-specified intervals to enable restarts (specified in :py:class:`tarexp.workflow.Workflow`). 

The persisted ledger for a complete run can be used to execute ``TARexp`` in *frozen* mode (:py:class:`tarexp.ledger.FrozenLedger`) 
where no batch selection, training, or scoring is done.  Frozen mode supports efficient testing of new components that 
do not change training or scoring, e.g., non-interventional stopping rules [1]_, effectiveness estimation methods, etc. 
Evaluating stopping rules for two-phase reviews also requires persisting scores of all documents at the end of each 
training round, an option the user can specify.    

.. seealso::
    .. [1] David D. Lewis, Eugene Yang, and Ophir Frieder. "Certifying One-Phase Technology-Assisted Reviews." 
           *Proceedings of the 30th ACM International Conference on Information & Knowledge Management*. 2021.
           `<https://arxiv.org/abs/2108.12746>`__

"""

from dataclasses import FrozenInstanceError
from collections import Counter
import numpy as np
from tarexp.base import Savable

class Ledger(Savable):
    
    def __init__(self, n_docs):
        super().__init__()
        # [i_round being annotated, label]
        self._record = np.full((n_docs, 2), np.nan) 
        self._n_rounds = -1

    def createControl(self, *args):
        if self._n_rounds != -1:
            raise AttributeError("Cannot create control set after initialization")
        self._n_rounds = -2
        self.annotate(*args)

    def annotate(self, *args):
        if len(args) == 1 and isinstance(args[0], dict):
            new_annotations = args[0]
        elif len(args) == 2:
            assert len(args[0]) == len(args[1]), "Mismatch number of document id and labels."
            new_annotations = dict(zip(*args))

        assert np.isnan(self._record[list(new_annotations.keys()), 0]).all(), \
               "All document annotating should not be annotated before"

        self._n_rounds += 1
        for doc_id, label in new_annotations.items():            
            self._record[doc_id] = (self._n_rounds, label)
        return len(new_annotations)

    def getReviewedIds(self, round: int):
        return np.where(self._record[:, 0] == round)[0]

    @property
    def control_mask(self):
        return self._record[:, 0] == -1

    @property
    def n_rounds(self):
        return self._n_rounds

    @property
    def n_docs(self):
        return self._record.shape[0]

    @property
    def n_annotated(self):
        return self.annotated.sum()
    
    @property
    def n_pos_annotated(self):
        return int(np.nan_to_num(self.annotation).sum())
    
    @property
    def n_neg_annotated(self):
        return self.n_annotated - self.n_pos_annotated

    @property
    def annotated(self):
        return ~np.isnan(self.annotation)

    @property
    def annotation(self):
        r = self._record[:, 1].copy()
        r[self.control_mask] = np.nan # remove control documents
        return r

    @property
    def isDone(self):
        return all(self.annotated)

    def getAnnotationCounts(self):
        return [
            Counter(dict(zip(*np.unique(self._record[ self._record[:, 0] == r ][:, 1], return_counts=True))))
            for r in range(self.n_rounds)
        ]
    
    def freeze(self):
        return FrozenLedger(self)
    
    def freeze_at(self, round: int):
        dup = self.freeze()
        dup._record.flags.writeable = True
        to_remove = dup._record[:, 0] > round
        dup._record[to_remove] = np.nan
        dup._record.flags.writeable = False
        return dup

class FrozenLedger(Ledger):

    def __init__(self, org_ledger: Ledger):
        self._record = org_ledger._record.copy()
        self._record.flags.writeable = False
    
    def annotate(self, *args, **kwargs):
        raise FrozenInstanceError

    @property
    def n_rounds(self):
        return int(np.nan_to_num(self._record[:, 0]).max())
    
