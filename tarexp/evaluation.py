"""Consistent implementation of effectiveness metrics, including tricky issues like tiebreaking is critical to TAR experiments. 
This is true both for evaluation, and because stopping rules may incorporate effectiveness estimates based on small samples. 
We provide all metrics from the open source package |ir-measures|_ through the :py:meth:`tarexp.workflow.Workflow.getMetrics` method. 
Metrics are computed on both the full collection and unreviewed documents to support both finite population and generalization 
perspectives. 

In addition to standard IR metrics, ``TARexp`` implements *OptimisticCost* (:py:class:`tarexp.evaluation.OptimisticCost`) to 
support the idealized end-to-end cost analysis for TAR proposed in  Yang *et al.* [1]_. Such analysis requires specifying a 
target recall and a cost structure associated with the TAR process. ``TARexp`` also provides helper functions for plotting 
cost dynamics graphs (:py:mod:`tarexp.helper.plotting`). 

.. |ir-measures| replace:: ``ir-measures``
.. _ir-measures: https://ir-measur.es/en/latest/

.. seealso::
    .. [1] Eugene Yang, David D. Lewis, and Ophir Frieder. 
           "On minimizing cost in legal document review workflows." 
           *Proceedings of the 21st ACM Symposium on Document Engineering*. 2021.
           `<https://arxiv.org/abs/2106.09866>`__

"""


from __future__ import annotations
from dataclasses import dataclass
from typing import Dict

import pandas as pd
import numpy as np
import ir_measures

from tarexp.ledger import Ledger
from tarexp.util import getOneDimScores

@dataclass(frozen=True)
class MeasureKey:
    """
    Hashable key for evaluation metric. 
    """
    measure: str | ir_measures.measures.Measure = None
    """Name of the measurement. """

    section: str = 'all'
    """Part of the collection that the evaluation is measured on. Can be but not limited to "all", "known" (reviewed 
        documents), etc. 
    """

    target_recall: float = None
    """The recall target. Can be ``None`` depending on whether the measure requires one. """

    def __hash__(self):
        return hash(repr(self))
    
@dataclass(frozen=True)
class CostCountKey(MeasureKey):
    """Hashable key for the recording count of the documents. 

    Attributes
    ----------
    measure
        Name of the measurement. 
    
    section
        Part of the collection that the evaluation is measured on. Can be but not limited to "all", "known" (reviewed 
        documents), etc. 
    
    target_recall
        The recall target. Can be ``None`` depending on whether the measure requires one. 

    """
    label: bool = None
    """The ground truth label that the measure is counting."""
    
    def __post_init__(self):
        object.__setattr__(self, 'measure', f"Count({self.label})")

@dataclass(frozen=True)
class OptimisticCost(MeasureKey):
    """Optimistic Cost
    
    The cost measure that records the total cost of reviewing documents in both first and (an optimal) second phase 
    review workflow. 
    Please refer to  Yang *et al.* [1]_ for further details.

    Attributes
    ----------
    measure
        Name of the measurement. 
    
    section
        Part of the collection that the evaluation is measured on. Can be but not limited to "all", "known" (reviewed 
        documents), etc. 
    
    target_recall
        The recall target. Can be ``None`` depending on whether the measure requires one. 
    """
    cost_structure: tuple[float, float, float, float] = None
    """Four-tuple cost structure. The elements of the tuple are the unit cost of reviewing positive and negative 
    documents in the first phase and positive and negative ones in the second phase respectively. 
    """

    def __post_init__(self):
        object.__setattr__(self, 'measure', f"OptimisticCost{repr(self.cost_structure)}")

    def __call__(self, *args):
        if len(args) == 1 and isinstance(args[0], pd.DataFrame):
            return OptimisticCost.calc_all([self], args[0])[self]
        elif len(args) == len(self.cost_structure):
            return sum([ 
                num*ucost for num, ucost in zip(args, self.cost_structure) 
            ])
    
    @staticmethod
    def calc_all(measures, df):
        """Static method for calculating multiple :py:class:`~OptimisticCost` measures given a cost Pandas DataFrame. 

        The dataframe should contain "query_id", "iteration", "relevance" (as ground truth), "score", "control" (boolean
        values of whether the document is in the control set), "known" (whether the document is reviewed) as columns and 
        all documents in the collection as rows.
        This dataframe is similar to the one used in ``ir_measures.calc_all()``. 

        This method is also used internally by :py:func:`~evaluate`.

        Parameters
        ----------
        measures
            A list of :py:class:`~OptimisticCost` instance for calculation.

        df
            The cost Pandas DataFrame 
        
        Returns
        -------
        dict
            A Python dictionary with key as the measures and values as the measurement values. 
            The count of each section given the all recall targets provided in the ``measures`` argument would also be 
            returned as auxiliary information.  
        """
        assert all([ isinstance(m, OptimisticCost) for m in measures ])
        all_recall_targets = set([ m.target_recall for m in measures ])
        
        df = df.assign(
            adjscore=((df.relevance-0.5)*np.inf*df.known).fillna(0) + df.score 
        ).sort_values('adjscore', ascending=False)
        
        ret = {}
        for tr in all_recall_targets:
            # oracle reviewed df
            ordf_above = df.iloc[: (df.relevance.cumsum() > df.relevance.sum()*tr).values.argmax() + 1]
            ordf_below = df.iloc[(df.relevance.cumsum() > df.relevance.sum()*tr).values.argmax() + 1:]

            ret[CostCountKey(target_recall=tr, section='known', label=True)] = (df.known & df.relevance).sum()
            ret[CostCountKey(target_recall=tr, section='known', label=False)] = (df.known & ~df.relevance).sum()
            ret[CostCountKey(target_recall=tr, section='unknown-above-cutoff', label=True)] = (~ordf_above.known & ordf_above.relevance).sum()
            ret[CostCountKey(target_recall=tr, section='unknown-above-cutoff', label=False)] = (~ordf_above.known & ~ordf_above.relevance).sum()
            ret[CostCountKey(target_recall=tr, section='unknown-below-cutoff', label=True)] = (~ordf_below.known & ordf_below.relevance).sum()
            ret[CostCountKey(target_recall=tr, section='unknown-below-cutoff', label=False)] = (~ordf_below.known & ~ordf_below.relevance).sum()
        for m in measures:
            ret[m] = m(*[ 
                ret[CostCountKey(target_recall=m.target_recall, section=sec, label=label)]
                for sec in ['known', 'unknown-above-cutoff'] 
                for label in [True, False] 
            ])
        return ret

def evaluate(labels, ledger: Ledger, score, measures) -> Dict[MeasureKey, int | float]:
    """Evaluate TAR run based on a given :py:class:`tarexp.ledger.Ledger`.

    This function calculates the evaluation metrics based on the provided Ledger.
    The measures are evaluated on the **last round** recorded in the ledger. 
    If inteded to caculate metrics on past rounds, please provide a ledger that only contains information up to the 
    round the user is intended to evaluate by using :py:meth:`tarexp.ledger.Ledger.freeze_at`. 

    It serves as a catch-all function for all evaluation metrics ``TARexp`` supports, including all measurements in
    |ir-measures|_ and :py:class:`~OptimisticCost`. 
    Future addtional of the supported evaluation metrics should also be added to this function for completeness. 

    Parameters
    ----------
    labels
        The ground-truth labels of the documents. This is different from the labels recorded in the Ledger which is the 
        review results (not necessarily the groun-truth if not using a :py:class:`tarexp.components.labeler.PerfectLabeler`)
    
    ledger
        The ledger the recorded the progress. 
    
    score
        A list of the document scores. 
    
    measures
        A list of :py:class:`~MeasureKey` instance, the name of the measurments supported in |ir-measures|_, or 
        |ir-measures|_ measurement object (such as ``ir_measures.P@10``). 
    
    Returns
    -------
    dict[:py:class:`~MeasureKey`, int|float]
        A Python dictionary of keys as instances of :py:class:`~MeasureKey` and values as the corresponding measurement
        values.
    """
    df = pd.DataFrame({
        'query_id': '0', 
        'iteration': str(ledger.n_rounds),
        'relevance': np.asanyarray(labels).astype(int), 
        'score': getOneDimScores(score),
        'control': ledger.control_mask,
        'known': ledger.annotated
    })
    df = df.assign(doc_id = df.index.astype(str))

    irms_measures, other_measures = [], []
    for m in measures:
        try: 
            irms_measures.append(ir_measures.parse_measure(m))
        except (NameError, TypeError):
            other_measures.append(m)

    ret = {}
    if len(irms_measures) > 0:
        ret.update({ 
            MeasureKey(measure=k, section=sec): v 
            for sec, d in zip(('all', 'unknown'), (df, df[df.known], df[~df.known]))
            for k, v in ir_measures.calc_aggregate(irms_measures, d, d).items() 
        })
    for cls in set([ m.__class__ for m in other_measures ]):
        ret.update(cls.calc_all([m for m in measures if isinstance(m, cls)], df))
    return ret
