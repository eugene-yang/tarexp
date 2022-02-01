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
    
