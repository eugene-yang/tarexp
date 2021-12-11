import numpy as np
from .base import Component
from ..ledger import Ledger

"""
Note:
Complex stopping rule could involve additional computation that leverages 
other components with builtin methods. `Control` will be able to facilatate that
but relies on user to behave and not abusing it. 
We could make it read-only but might be very messy and ambiguous. 
"""

class StoppingRule(Component):
    def __init__(self):
        super().__init__()
    
    def checkStopping(self, ledger: Ledger, *args, **kwargs) -> bool:
        raise NotImplementedError
        
    
class NullStoppingRule(StoppingRule):
    def checkStopping(self, *args, **kwargs):
        return False


class FixedRoundStoppingRule(StoppingRule):
    def __init__(self, max_round, *args, **kwargs):
        assert max_round > 0
        self._max_round = max_round

    def checkStopping(self, ledger, *args, **kwargs):
        return ledger.n_rounds >= self._max_round


# TODO: convert them to stopping rules ------

# R1: knee method
def stopping_knee(pos_found, batchsize=200, **kwargs):
    pos_found = np.asanyarray(pos_found) - 1
    for s in range(len(pos_found)):
        rho_s = -1
        for i in range(s):
            rho = (pos_found[i]/(i+1)) / ((1+pos_found[s]-pos_found[i])/(s-i+1))
            rho_s = max(rho_s, rho)
        if rho_s >= 156 - min(pos_found[s], 150):
            # return s, rho_s
            return s
    # return -1, np.inf
    return len(pos_found)-1

# R2: budget method
def stopping_budget(pos_found, batchsize=200, **kwargs):
    pos_found = np.asanyarray(pos_found) - 1
    for s in range(len(pos_found)):
        rho_s = -1
        for i in range(s):
            rho = (pos_found[i]/(i+1)) / ((1+pos_found[s]-pos_found[i])/(s-i+1))
            rho_s = max(rho_s, rho)
        if (rho_s >= 6 and batchsize*i+1 >= 10*using_rel.shape[0] / pos_found[i]) or \
           s*batchsize+1 >= using_rel.shape[0]*0.75:
            # return s, rho_s
            return s
    # return -1, np.inf
    return len(pos_found)-1

# R3: review half
def stopping_half(pos_found, batchsize=200, **kwargs):
    half = int(np.ceil((using_rel.shape[0]-1) / batchsize))
    return half if len(pos_found) > half else len(pos_found) - 1

# R8: Batch precision cutoff(O1)
def stopping_precision(pos_found, batchsize=200, cutoff=5/200, slack=1, **kwargs):
    pos_found = np.asanyarray(pos_found)
    bpre = (pos_found[1:] - pos_found[:-1]) / batchsize
    return ((bpre <= cutoff).cumsum() >= slack).argmax() + 1

# R10: Elusion test
def stopping_elusion(pos_found, batchsize=200, cutoff=1e-4, slack=1, **kwargs):
    npos = npos_20sub[ pos_found.index[0][0] ]
    N = using_rel.shape[0]
    unreviewed = np.linspace(N-1, N-1-(200*(len(pos_found)-1)), len(pos_found))
    pos_found = np.asanyarray(pos_found)
    elu = (npos-pos_found) / unreviewed
    return ((elu <= cutoff).cumsum() >= slack).argmax() + 1

# R11: G&M Trec R_hat + 2399 
def stopping_gmtrec(pos_found, batchsize=200, **kwargs):
    return ((np.arange(len(pos_found))*200 + 1) >= (1.2*pos_found + 2399)).argmax() + 1

# and quant + quant CI

# CMH heuristics should put in CMH module alone with its interventional rule