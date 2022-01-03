from __future__ import annotations
from dataclasses import dataclass
from typing import Dict

import pandas as pd
import numpy as np
import ir_measures

from tarpy.ledger import Ledger
from tarpy.util import getOneDimScores

@dataclass(frozen=True)
class MeasureKey:
    measure: str | ir_measures.measures.Measure = None
    section: str = 'all'
    target_recall: float = None

    def __hash__(self):
        return hash(repr(self))
    
@dataclass(frozen=True)
class CostCountKey(MeasureKey):
    label: bool = None
    
    def __post_init__(self):
        object.__setattr__(self, 'measure', f"Count({self.label})")

@dataclass(frozen=True)
class OptimisticCost(MeasureKey):
    """Optimistic cost measures
    These cost measures relies on an oracle for the optimal cutoff of the rank. 
    """
    cost_structure: tuple[float, float, float, float] = None

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

            ret[CostCountKey(target_recall=tr, section='training', label=True)] = (df.known & df.relevance).sum()
            ret[CostCountKey(target_recall=tr, section='training', label=False)] = (df.known & ~df.relevance).sum()
            ret[CostCountKey(target_recall=tr, section='unknown-above-cutoff', label=True)] = (~ordf_above.known & ordf_above.relevance).sum()
            ret[CostCountKey(target_recall=tr, section='unknown-above-cutoff', label=False)] = (~ordf_above.known & ~ordf_above.relevance).sum()
            ret[CostCountKey(target_recall=tr, section='unknown-below-cutoff', label=True)] = (~ordf_below.known & ordf_below.relevance).sum()
            ret[CostCountKey(target_recall=tr, section='unknown-below-cutoff', label=False)] = (~ordf_below.known & ~ordf_below.relevance).sum()
        for m in measures:
            ret[m] = m(*[ 
                ret[CostCountKey(target_recall=m.target_recall, section=sec, label=label)]
                for sec in ['training', 'unknown-above-cutoff'] 
                for label in [True, False] 
            ])
        return ret

def evaluate(labels, ledger: Ledger, score, measures) -> Dict[MeasureKey, int | float]:
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

    ret = { 
        MeasureKey(measure=k, section=sec): v 
        for sec, d in zip(('all', 'unknown'), (df, df[~df.known]))
        for k, v in ir_measures.calc_aggregate(irms_measures, d, d).items() 
    }
    for cls in set([ m.__class__ for m in other_measures ]):
        ret.update(cls.calc_all([m for m in measures if isinstance(m, cls)], df))
    return ret
