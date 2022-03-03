"""``TARexp`` implements algorithms via *components*. A component is an object that is declared to serve one or more *roles* 
in a workflow, e.g. the stopping rule, the training batch sampler, the ranker, or the labeler. Components communicate only through 
the workflow. The association of components with multiple roles is important when implementing algorithms where, 
for instance, the stopping rule interacts tightly with a particular batch selection method (e.g. AutoStop [1]_).

.. TODO: talk about basic begin, reset interface here

.. seealso::
    .. [1] Dan Li, and Evangelos Kanoulas. 
           "When to stop reviewing in technology-assisted reviews: Sampling from an adaptive distribution to estimate residual relevant documents." 
           *ACM Transactions on Information Systems (TOIS)* 38.4 (2020): 1-36.
           `<https://dl.acm.org/doi/10.1145/3411755>`__

"""


from tarexp.component.base import Component, asComponent, CombinedComponent, combine
from tarexp.component.ranker import Ranker, SklearnRanker 
from tarexp.component.labeler import Labeler, PerfectLabeler
from tarexp.component.sampler import Sampler, \
                                     RandomSampler, \
                                     UncertaintySampler, \
                                     RelevanceSampler
from tarexp.component.stopping import StoppingRule, \
                                      NullStoppingRule, \
                                      FixedRoundStoppingRule, \
                                      KneeStoppingRule, \
                                      BudgetStoppingRule, \
                                      ReviewHalfStoppingRule, \
                                      BatchPrecStoppingRule, \
                                      Rule2399StoppingRule, \
                                      QuantStoppingRule, \
                                      CHMHeuristicsStoppingRule