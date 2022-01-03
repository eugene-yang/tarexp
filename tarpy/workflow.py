from __future__ import annotations
from typing import Dict, List
from warnings import warn
from collections import OrderedDict

from ir_measures.measures import Measure

from pathlib import Path
import numpy as np

from tarpy.base import Savable
from tarpy.component import Component
from tarpy.ledger import FrozenLedger, Ledger
from tarpy.dataset import Dataset
from tarpy.evaluation import MeasureKey, OptimisticCost, evaluate

from tarpy.util import saveObj, readObj

def checkAutoRoles(component: Component):
    return component.hasRanker and \
           component.hasStoppingRule and \
           component.hasLabeler and \
           component.hasSampler


class Workflow(Savable):

    def __init__(self, dataset: Dataset, component: Component, 
                       saved_score_limit: int = -1,
                       random_seed: int = None, 
                       resume: bool = False, **kwargs):
        super().__init__()
        if not checkAutoRoles(component):
            warn("Input component does not have all essential roles. "
                 "Require manual intervention during iterations to avoid "
                 "infinite loop.")
        
        self._random = np.random.RandomState(random_seed)
        self.saved_score_limit = saved_score_limit

        self._component = component
        self._dataset = dataset
        self._ledger = Ledger(self._dataset.n_docs)
        self._saved_scores = OrderedDict()

        if not resume and self._component is not None:
            self._component.begin(dataset=self._dataset, 
                                  random_seed=random_seed, 
                                  **kwargs)
        self._stopped = None
    
    @property
    def _saving_attrs(self):
        return ['_random', 'saved_score_limit', '_saved_scores']
    
    @property
    def ledger(self):
        return self._ledger
    
    @property
    def component(self):
        return self._component

    @property
    def dataset(self):
        return self._dataset
    
    @property
    def n_rounds(self):
        return self.ledger.n_rounds

    @property
    def isStopped(self):
        # cache the decision given the same round
        if self._stopped is None or self._stopped[0] != self.n_rounds:
            self._stopped = (self.n_rounds, 
                             self.component.checkStopping(self.ledger.freeze()))
        return self._stopped[1] or self.ledger.isDone

    @property
    def latest_scores(self):
        return self._saved_scores[next(reversed(self._saved_scores))]

    def _saveScores(self, scores, overwrite=False):
        if not overwrite and self.n_rounds in self._saved_scores:
            raise AttributeError(f"Already saved scores at round {self.n_rounds}.")
        self._saved_scores[self.n_rounds] = scores

        if self.saved_score_limit > 0 and \
           len(self._saved_scores) > self.saved_score_limit:
            self._saved_scores.pop(next(iter(self._saved_scores)))

    def __next__(self):
        if self.isStopped: # shoudn't invoke stopping rule again
            raise StopIteration
        self.step()
        return self.ledger.freeze()
    
    def __iter__(self):
        return self
    
    def save(self, output_dir, with_component=True, overwrite=False):
        output_dir = Path(output_dir) / f"it_{self.ledger.n_rounds}"
        if output_dir.exists():
            if not output_dir.is_dir():
                raise NotADirectoryError(f"Path {output_dir} is not a directory.") 
            if not overwrite:
                raise FileExistsError(f"Path {output_dir} already exists.")
        output_dir.mkdir(parents=True, exist_ok=True)

        saveObj({
            **{att: getattr(self, att) for att in self._saving_attrs},
            'dataset_label_idn': self.dataset.identifier,
            "current_round": self.ledger.n_rounds,
            "component.class": self._component.__class__,
            "ledger.class": self._ledger.__class__,
        }, output_dir / "workflow_config.pgz")

        if with_component:
            self._component.save(output_dir) 
        self._ledger.save(output_dir)

        return output_dir
    
    @classmethod
    def load(cls, saved_path, dataset: Dataset, force=False):
        saved_path = Path(saved_path)
        if not saved_path.exists() or not saved_path.is_dir():
            raise FileNotFoundError(f"Cannot find directory {saved_path}.")
        
        if not (saved_path / "workflow_config.pgz").exists():
            # find the last iteration
            last_iter = max([ int(d.name.split("_")[1]) 
                              for d in saved_path.glob("it_*") ])
            print(f"Found saved info for iteration {last_iter}.")
            saved_path = saved_path / f"it_{last_iter}"

        saved_attrs = readObj(saved_path / "workflow_config.pgz", dict)
        ledger = saved_attrs['ledger.class'].load(saved_path)

        replay_only = issubclass(cls, WorkflowReplay)
        
        # sanity checks
        if not force and not replay_only:
            assert saved_attrs['current_round'] == ledger.n_rounds
            assert saved_attrs['dataset_label_idn'] == dataset.identifier

        if not replay_only:
            components = saved_attrs['component.class'].load(saved_path)
            resume_workflow = cls(dataset, components, resume=True)
            resume_workflow._ledger = ledger
        else:
            resume_workflow = cls(dataset, ledger.freeze())
        
        for att in resume_workflow._saving_attrs:
            setattr(resume_workflow, att, saved_attrs[att])
        
        return resume_workflow
    
    def step(self, step=False):
        # starts by checking stopping condition because Control could have 
        # changed the state
        raise NotImplementedError
        
    def getMetrics(self, measures: List[OptimisticCost | Measure | str]) -> Dict[MeasureKey, int | float]:
        raise NotImplementedError
    
    def makeReplay(self):
        raise NotImplementedError

class WorkflowReplay(Workflow):
    def __init__(self, dataset: Dataset, ledger: FrozenLedger,
                       saved_scores: OrderedDict = None,
                       random_seed: int = None,   
                       **kwargs):
        self._random = np.random.RandomState(random_seed)

        self._dataset = dataset
        self._ledger = ledger
        self._saved_scores = saved_scores # rely on load to replace it

        self._replay_round = -1

    @property
    def n_rounds(self):
        return self._replay_round
    
    @property
    def ledger(self):
        return self._ledger.freeze_at(self.n_rounds)

    @property
    def isStopped(self):
        return self._replay_round >= self._ledger.n_rounds
    
    def step(self):
        if self.isStopped:
            return 
        self._replay_round += 1

    def save(self, *args, **kwargs):
        raise NotImplemented("Workflow replay is not savable")
        

class OnePhaseTARWorkflow(Workflow):

    def __init__(self, dataset: Dataset, component: Component, 
                 seed_doc: list = [], batch_size: int = 200,
                 control_set_size: int = 0, 
                 saved_score_limit: int = -1,
                 random_seed: int = None, 
                 **kwargs):
        super().__init__(dataset, component, saved_score_limit, 
                         random_seed, **kwargs)
        if not self.component.hasRanker:
            raise ValueError("Must have a ranker in the component for "
                             f"{self.__class__.__name__}")
        
        self.batch_size = batch_size

        control_set = []
        if control_set_size > 0:
            control_set = self._random.choice(self.dataset.n_docs, 
                                              size=control_set_size, 
                                              replace=False)
            self.ledger.createControl(control_set,
                                      self.component.labelDocs(control_set))

        self.review_candidates = seed_doc

    @property
    def _saving_attrs(self):
        return super()._saving_attrs + \
               ['batch_size', 'review_candidates']

    def step(self, force=False):
        if self.isStopped and not force:
            return 
        self.ledger.annotate(self.review_candidates, 
                             self.component.labelDocs(self.review_candidates))
        self.component.trainRanker(*self.dataset.getTrainingData(self.ledger))
        self._saveScores(
            self.component.scoreDocuments(self.dataset.getAllData())
        )
        self.review_candidates = self.component.sampleDocs(
            self.batch_size, self.ledger, self.latest_scores
        )

    def getMetrics(self, measures, labels=None):
        if labels is None:
            if not self.dataset.hasLabels:
                raise ValueError("Labels are not provided.")
            labels = self.dataset.labels
        return evaluate(labels, self.ledger, self.latest_scores, measures)

    def makeReplay(self):
        return OnePhaseTARWorkflowReplay(self.dataset, self.ledger.freeze(), self._saved_scores)

# inheritance order matters here -- wants MRO to take replay first
class OnePhaseTARWorkflowReplay(WorkflowReplay, OnePhaseTARWorkflow):

    def __init__(self, dataset: Dataset, ledger: FrozenLedger, 
                       saved_scores=None):
        super().__init__(dataset, ledger, saved_scores)

        self._last_review_candidates = None

    @property
    def latest_scores(self):
        if self._replay_round not in self._saved_scores:
            raise KeyError(f"Scores of iteration {self._replay_round} was not saved.")
        return self._saved_scores[self._replay_round]
    
    @property
    def review_candidates(self):
        if self.isStopped:
            return self._last_review_candidates
        return self.ledger.getReviewedIds(self._replay_round+1)

    @review_candidates.setter
    def review_candidates(self, candidates):
        self._last_review_candidates = candidates


class TwoPhaseTARWorkflow(OnePhaseTARWorkflow):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.component.checkRole('after_stopping'):
            raise ValueError("Two phase workflow requires role `after_stopping`.")

    def step(self, *args, **kwargs):
        super().step(*args, **kwargs)
        if self.isStopped:
            self.component.after_stopping(
                self.ledger, self.component, self.dataset
            )