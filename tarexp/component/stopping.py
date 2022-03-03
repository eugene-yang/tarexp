"""Stopping rules are a particular focus of TAR research. 
"""

import numpy as np
from tarexp.component.base import Component
from tarexp.ledger import Ledger
from tarexp.util import getOneDimScores
# Note: not importing Workflow to avoid circular dependency

from scipy.stats import hypergeom 


def _inferBatchsize(ledger: Ledger, fast=True):
    if fast:
        return ledger.n_annotated // ledger.n_rounds    
    batch_size = None
    for r in range(1, ledger.n_rounds): # round 0 is the seed set
        current_bs = (ledger._record[:, 0] == r).sum()
        if batch_size is None:
            batch_size = current_bs
        assert batch_size == current_bs
    return batch_size


class StoppingRule(Component):
    def __init__(self, target_recall: float=None):
        super().__init__()
        self.target_recall = target_recall
    
    def checkStopping(self, ledger: Ledger, *args, **kwargs) -> bool:
        raise NotImplementedError
        
    
class NullStoppingRule(StoppingRule):
    def checkStopping(self, *args, **kwargs):
        return False


class FixedRoundStoppingRule(StoppingRule):
    def __init__(self, max_round, *args, **kwargs):
        super().__init__(**kwargs)
        assert max_round >= 0
        self.max_round = max_round

    def checkStopping(self, ledger, *args, **kwargs):
        return ledger.n_rounds >= self.max_round


class KneeStoppingRule(StoppingRule):
    """
    .. seealso::
        .. [1] Gordon V. Cormack, and Maura R. Grossman. "Engineering quality and reliability in technology-assisted review." 
               *Proceedings of the 39th International ACM SIGIR conference on Research and Development in Information Retrieval.* 2016.
               `<https://dl.acm.org/doi/10.1145/2911451.2911510>`__
    """

    def checkStopping(self, ledger: Ledger, *args, **kwargs) -> bool:
        if ledger.n_rounds < 1:
            return False

        pos_per_round = np.array([ c[1] if 1 in c else 0 for c in ledger.getAnnotationCounts() ])
        pos_found = pos_per_round.cumsum()
        
        rho_s = -1
        for i in range(ledger.n_rounds):
            rho = (pos_found[i]/(i+1)) / ((1+pos_found[-1]-pos_found[i])/(ledger.n_rounds-i))
            rho_s = max(rho_s, rho)

        return rho_s >= 156 - min(pos_found[-1], 150)


class BudgetStoppingRule(StoppingRule):
    """
    .. seealso::
        .. [2] Gordon V. Cormack, and Maura R. Grossman. "Engineering quality and reliability in technology-assisted review." 
               *Proceedings of the 39th International ACM SIGIR conference on Research and Development in Information Retrieval.* 2016.
               `<https://dl.acm.org/doi/10.1145/2911451.2911510>`__
    """

    def checkStopping(self, ledger: Ledger, *args, **kwargs) -> bool:
        if ledger.n_rounds < 1:
            return False
            
        batchsize = _inferBatchsize(ledger)
        pos_per_round = np.array([ c[1] if 1 in c else 0 for c in ledger.getAnnotationCounts() ])
        pos_found = pos_per_round.cumsum()
        
        rho_s = -1
        for i in range(ledger.n_rounds):
            rho = (pos_found[i]/(i+1)) / ((1+pos_found[-1]-pos_found[i])/(ledger.n_rounds-i))
            rho_s = max(rho_s, rho)

        return  (rho_s >= 6 and batchsize*i+1 >= 10*ledger.n_docs / pos_found[i]) or \
                (ledger.n_annotated >= ledger.n_docs*0.75)


class ReviewHalfStoppingRule(StoppingRule):
    def checkStopping(self, ledger: Ledger, *args, **kwargs) -> bool:
        return ledger.n_annotated >= ledger.n_docs // 2


class BatchPrecStoppingRule(StoppingRule):
    def __init__(self, prec_cutoff=5/200, slack=1):
        super().__init__()
        self.prec_cutoff = prec_cutoff
        self.slack = slack
    
    def checkStopping(self, ledger: Ledger, *args, **kwargs) -> bool:
        bprec = np.array([ batch[1] / sum(batch.values()) for batch in ledger.getAnnotationCounts() ])
        counter = 0
        for prec in bprec:
            counter = (counter+1) if prec <= self.prec_cutoff else 0
            if counter >= self.slack:
                return True
        return False


class Rule2399StoppingRule(StoppingRule):
    def checkStopping(self, ledger: Ledger, *args, **kwargs) -> bool:
        return ledger.n_annotated >= 1.2*ledger.n_pos_annotated + 2399


class QuantStoppingRule(StoppingRule):
    """
    .. seealso::
        .. [3] Eugene Yang, David D. Lewis, and Ophir Frieder. "Heuristic stopping rules for technology-assisted review." 
               *Proceedings of the 21st ACM Symposium on Document Engineering.* 2021.
               `<https://arxiv.org/abs/2106.09871>`__
    """
    def __init__(self, target_recall: float, nstd: float = 0):
        super().__init__(target_recall=target_recall)
        self.nstd = nstd
    
    def checkStopping(self, ledger: Ledger, workflow, **kwargs) -> bool:
        if ledger.n_rounds < 2:
            return False

        scores = getOneDimScores(workflow.latest_scores)
            
        assert (scores <= 1).all() and (scores >= 0).all(), \
                "Scores have to be probabilities to use Quant Rule."

        # `ps` stands for probability sum
        unknown_ps = scores[ ~ledger.annotated ].sum()
        known_ps = scores[ ledger.annotated ].sum()
        est_recall = (known_ps) / (known_ps + unknown_ps)
        if self.nstd == 0:
            return est_recall >= self.target_recall
        
        prod = scores * (1-scores)
        all_var = prod.sum()
        unknown_var = prod[ ~ledger.annotated ].sum()

        est_var = (known_ps**2 / (known_ps + unknown_ps)**4 * all_var) + (1 / (known_ps + unknown_ps)**2 * (all_var-unknown_var))
        
        return est_recall - self.nstd*np.sqrt(est_var) >= self.target_recall

class CHMHeuristicsStoppingRule(StoppingRule):
    """
    .. seealso::
        .. [4] Max W. Callaghan, and Finn Müller-Hansen. "Statistical stopping criteria for automated screening in systematic reviews." 
               *Systematic Reviews 9.1* (2020): 1-14.
               `<https://pubmed.ncbi.nlm.nih.gov/33248464/>`__
    """
    def __init__(self, target_recall: float, alpha=0.05):
        super().__init__(target_recall=target_recall)
        self.alpha = alpha
    
    def checkStopping(self, ledger: Ledger, *args, **kwargs) -> bool:
        if ledger.n_rounds < 2:
            return False

        counts = ledger.getAnnotationCounts()
        pos_found = np.array([ c[1] if 1 in c else 0 for c in counts ]).cumsum()
        annotated_cumsum = np.array([ sum(c.values()) for c in counts ]).cumsum()
        n_docs = ledger.n_docs
        
        for i in range(1, ledger.n_rounds):
            if hypergeom.cdf( pos_found[-1] - pos_found[i], # k
                              n_docs-annotated_cumsum[i], # N
                              int(pos_found[-1]/self.target_recall - pos_found[i]), # K_tar
                              annotated_cumsum[-1] - annotated_cumsum[i] # n
                            ) < self.alpha:
                return True
        return False

    # Note: Full CMH rule is actually a two-phase workflow where the poststopping does a random walk


