from dataclasses import dataclass
from multiprocessing import Value
from warnings import warn
from collections import OrderedDict

from pathlib import Path
import numpy as np

from tarpy.base import Savable
from tarpy.component import Component
from tarpy.ledger import FrozenLedger, Ledger
from tarpy.dataset import Dataset
from tarpy.evaluation import evaluate

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

        if not resume:
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
    
    def save(self, output_dir, overwrite=False):
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
            "current_round": self.ledger.n_rounds
        },output_dir / "workflow_config.pgz")

        # Might be better to put it in the abstract class
        self._component.save(output_dir / "components.pgz")
        self._ledger.save(output_dir / "ledger.pgz")
    
    @classmethod
    def load(cls, output_dir, dataset: Dataset, force=False):
        output_dir = Path(output_dir)
        if not output_dir.exists() or not output_dir.is_dir():
            raise FileNotFoundError(f"Cannot find directory {output_dir}.")
        
        if not (output_dir / "workflow_config.pgz").exists():
            # find the last iteration
            last_iter = max([ int(d.name.split("_")[1]) 
                              for d in output_dir.glob("it_*") ])
            print(f"Found saved info for iteration {last_iter}.")
            output_dir = output_dir / f"it_{last_iter}"

        saved_attrs = readObj(output_dir / "workflow_config.pgz", dict)
        components = readObj(output_dir / "components.pgz", Component)
        ledger = readObj(output_dir / "ledger.pgz", Ledger)

        # sanity checks
        if not force:
            assert saved_attrs['current_round'] == ledger.n_rounds
            assert saved_attrs['dataset_label_idn'] == dataset.identifier

        resume_workflow = cls(dataset, components, resume=True)
        resume_workflow._ledger = ledger
        for att in resume_workflow._saving_attrs:
            setattr(resume_workflow, att, saved_attrs[att])
        
        return resume_workflow
    
    def step(self, step=False):
        # starts by checking stopping condition because Control could have 
        # changed the state
        raise NotImplementedError
        
    def getMetrics(self):
        raise NotImplemented


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

        self._review_candidates = seed_doc

    @property
    def _saving_attrs(self):
        return super()._saving_attrs + \
               ['batch_size', '_review_candidates']

    def step(self, force=False):
        if self.isStopped and not force:
            return 
        self.ledger.annotate(self._review_candidates, 
                             self.component.labelDocs(self._review_candidates))
        self.component.trainRanker(*self.dataset.getTrainingData(self.ledger))
        self._saveScores(
            self.component.scoreDocuments(self.dataset.getAllData())
        )
        self._review_candidates = self.component.sampleDocs(
            self.batch_size, self.ledger, self.latest_scores
        )

    def getMetrics(self, measures, labels=None):
        if labels is None:
            if not self.dataset.hasLabels:
                raise ValueError("Labels are not provided.")
            labels = self.dataset.labels
        return evaluate(labels, self.ledger, self.latest_scores, measures)
    
    def overwriteCandidates(self, new_candidates):
        self._review_candidates = new_candidates


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