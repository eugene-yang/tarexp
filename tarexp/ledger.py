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

from __future__ import annotations
from dataclasses import FrozenInstanceError
from collections import Counter
from typing import Dict, List
import numpy as np
from tarexp.base import Savable

class Ledger(Savable):
    """A :py:class:`~Ledger` records the progress of a TAR run. 

    Specifically, we record (1) the review result of each document and (2) the round each document being reviewed.
    Control set documents are marked as reviewed at round `-1` and seed documents are at round `0`. 

    Parameters
    ----------
    n_docs
        Number of documents in the collection.
    """
    
    def __init__(self, n_docs: int):
        super().__init__()
        assert n_docs > 0
        # [i_round being annotated, label]
        self._record = np.full((n_docs, 2), np.nan) 
        self._n_rounds = -1

    def createControl(self, *args):
        """Create a control set
        Plese refer to :py:meth:`~annoate` for the argument documentation.
        """
        if self._n_rounds != -1:
            raise AttributeError("Cannot create control set after initialization")
        self._n_rounds = -2
        self.annotate(*args)

    def annotate(self, *args):
        """Record a round of annotation(review). 

        Parameters
        ----------
        Dictionary
            If only one positional argument is provided, it should be a dictionary with keys being the document id and 
            values being the corresponding labels.
        
        Pair
            If two positional arguments are provided, the first one is treated as the documents and the second one is
            the corresponding labels. The length of the two lists are required be the same. 
        
        Returns
        -------
        int
            Number of documents being annotated at this round.
        """
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

    def getReviewedIds(self, round: int) -> np.ndarray[int]:
        """Get the list of document id that are reviewed at ``round``."""
        return np.where(self._record[:, 0] == round)[0]

    @property
    def control_mask(self) -> np.ndarray[bool | np.nan]:
        """The mask of the size of the collection for the control documents."""
        return self._record[:, 0] == -1

    @property
    def n_rounds(self) -> int:
        """Number of rounds have executed"""
        return self._n_rounds

    @property
    def n_docs(self) -> int:
        """Number of docuemnts in the collection."""
        return self._record.shape[0]

    @property
    def n_annotated(self) -> int:
        """Total number of documents have been reviewed."""
        return self.annotated.sum()
    
    @property
    def n_pos_annotated(self) -> int:
        """Total number of documents have been labeled as positive (relevant)."""
        return int(np.nan_to_num(self.annotation).sum())
    
    @property
    def n_neg_annotated(self) -> int:
        """Total number of documents have been labeled as negative (non-relevant)."""
        return self.n_annotated - self.n_pos_annotated

    @property
    def annotated(self) -> np.ndarray[bool]:
        """The mask of the size of the collection for the annotated (reviewed) documents."""
        return ~np.isnan(self.annotation)

    @property
    def annotation(self) -> np.ndarray[bool | np.nan]:
        """List of the annotations. Documents that have not been reviewed are recorded as ``np.nan``. 
        Control documents are considered **not annotated**.
        """
        r = self._record[:, 1].copy()
        r[self.control_mask] = np.nan # remove control documents
        return r

    @property
    def isDone(self) -> bool:
        """Whether all documents have been reviewed (including control docuemnts.). """
        return all(self.annotated)

    def getAnnotationCounts(self) -> List[Dict[bool, int]]:
        """Get a list of dictionaries that records the number of positive and negative documents reviewed in each round."""
        return [
            Counter(dict(zip(*np.unique(self._record[ self._record[:, 0] == r ][:, 1], return_counts=True))))
            for r in range(self.n_rounds)
        ]
    
    def freeze(self) -> FrozenLedger:
        """Get a frozen version of the current ledger."""
        return FrozenLedger(self)
    
    def freeze_at(self, round: int) -> FrozenLedger:
        """Get a frozen version of the ledger at ``round``."""
        dup = self.freeze()
        dup._record.flags.writeable = True
        to_remove = dup._record[:, 0] > round
        dup._record[to_remove] = np.nan
        dup._record.flags.writeable = False
        return dup

class FrozenLedger(Ledger):
    """A frozen ledger prohibits the record being modified. 
    The underlying record (implemented as a numpy array) is locked as not writable. 
    
    All methods and properties of :py:class:`~Ledger` are supported except :py:meth:`tarexp.ledger.Ledger.annotate` which
    will raise a ``FrozenInstanceError`` when invoked. 
    """

    def __init__(self, org_ledger: Ledger):
        self._record = org_ledger._record.copy()
        self._record.flags.writeable = False
    
    def annotate(self, *args, **kwargs):
        """Not supported."""
        raise FrozenInstanceError

    @property
    def n_rounds(self):
        """Number of rounds this frozen ledger have recorded."""
        return int(np.nan_to_num(self._record[:, 0]).max())
    
