"""An instance of class :py:class:`~Workflow` executes the user's declarative specification of a TAR workflow.  
In doing so, it reaches out to :py:class:`tarexp.component.Component` for services specified in the declarative specification 
such as creating training batches, scoring and ranking the collection, and testing for stopping conditions.  

After an optional initial seed round where the user can specify a starting set of labeled training data, 
the workflow is executed as a sequence of training rounds. Each round consists of selecting a batch of 
training documents (using :py:class:`tarexp.component.Sampler`), looking up labels for those documents 
(using :py:class:`tarexp.component.Labeler`), training a model and scoring and ranking the collection 
documents (using :py:class:`tarexp.component.Ranker`).

``TARexp`` supports specifications of both one and two-phase TAR workflows, as described in Yang *et al.* [1]_. 
One-phase workflows (:py:class:`~OnePhaseTARWorkflow`) can be run for a fixed number of training rounds, or 
until all documents have been reviewed.  Two-phase reviews also use a stopping rule to determine when to end training, 
but then follow that by ranking the collection with the final trained model and reviewing to a statistically determined cutoff.

:py:class:`~Workflow` is implemented as a Python iterator, allowing procedures defined outside the workflow 
to execute at each round. The iterator yields a :py:class:`tarexp.ledger.FrozenLedger`. 
The user can define a custom per-round evaluation process or record information for later analysis. 

:py:class:`~WorkflowReplay` is a special kind of workflow that replays an existing wokrflow record. 
The replay executes the original TAR run without changing the documents being reviewed at each round. 
It can be used to gather more information throughout the process and testing other components based on
an exisiting TAR run such as stopping rule. 

.. seealso::
    .. [1] Eugene Yang, David D. Lewis, and Ophir Frieder. 
           "On minimizing cost in legal document review workflows." 
           *Proceedings of the 21st ACM Symposium on Document Engineering*. 2021.
           `<https://arxiv.org/abs/2106.09866>`__

"""

from __future__ import annotations
from typing import Dict, List
from warnings import warn
from collections import OrderedDict
from shutil import rmtree

# from ir_measures.measures import Measure
import ir_measures

from pathlib import Path
import numpy as np

from tarexp.base import Savable
from tarexp.component.base import Component
from tarexp.ledger import FrozenLedger, Ledger
from tarexp.dataset import Dataset
from tarexp.evaluation import MeasureKey, OptimisticCost, evaluate

from tarexp.util import saveObj, readObj

def _checkAutoRoles(component: Component):
    return component.hasRanker and \
           component.hasStoppingRule and \
           component.hasLabeler and \
           component.hasSampler


class Workflow(Savable):
    """Meta workflow class. 

    The meta workflow class provides essential interface and bookkeeping for most workflows.
    It also implement the iterator interface which yields a :py:class:`tarexp.ledger.FrozenLedger` in each round. 
    The state of a workflow instance is stored in the :py:attr:`~ledger`. 

    All workflow inheriting this class should implement the :py:meth:`~step` method which defines the TAR process at each step. 
    Any initialization of the workflow, including sampling the control set, should be implemented in the :py:meth:`~__init__` method. 
    :py:meth:`~getMetrics` and :py:meth:`~makeReplay` are optional. 

    .. warning::
        Workflow initialization raises warnning when the sepcifying component has all four essential roles for running 
        a TAR experiment (ranker, stopping rule, labeler, and sampler). 
        However, workflow can execute without one or multiple of them but simply skipping such process **silently**. 
        User need to manually perform these processes in the iterator block or after invoking :py:meth:`~step` to ensure
        the workflow executes correctly.

    Parameters
    ----------
    dataset
        Textual dataset the workflow will search on. When running experiments, it should contain gold labels but not strictly 
        required (might raises exceptions depending on the labeler). User could manually provide labels to the workflow in
        each step manually. 

    component
        A combined component that contains ideally all 4 essential roles. Components can be combined by using 
        :py:func:`tarexp.component.base.combine`. 

    max_round_exec
        Maximum round the workflow will execute before the stopping rule suggests stopping. ``-1`` indicates no limit. 
        Default is ``-1``. 

    saved_score_limit
        Number of set of documents' score from each round would be stored in memory. Number smaller than 0 indicates 
        no limitation. This would also affect the saved checkpoint, so for full replay capability, the value should be < 0. 
        Default is ``-1``. 0 is not allowed. 

    saved_checkpoint_limit
        Number of workflow checkpoints would be stored on disk in the output directory. If the limit is reached, older 
        checkpoint will be deleted **silently**. Default ``2`` (fewer than 2 is not recommanded for stability as the latest 
        checkpoint could be incomplete if the failure happens during saving). 
    
    random_seed
        Random seed for any randomization process (if any) used in the workflow. Default ``None``. 

    resume
        Not initializing the components if ``True``. Used internally by the :py:meth:`~load` method. Default ``False``. 
    
    """

    def __init__(self, dataset: Dataset, component: Component, 
                       max_round_exec: int = -1, 
                       saved_score_limit: int = -1,
                       saved_checkpoint_limit: int = 2,
                       random_seed: int = None, 
                       resume: bool = False, **kwargs):
        super().__init__()
        if not _checkAutoRoles(component):
            warn("Input component does not have all essential roles. "
                 "Require manual intervention during iterations to avoid "
                 "infinite loop.")
        
        assert saved_score_limit != 0, "Should be either < 0 (no limit) or at least 1."

        self._random = np.random.RandomState(random_seed)
        self.saved_score_limit = saved_score_limit
        self.saved_checkpoint_limit = saved_checkpoint_limit
        self.max_round_exec = max_round_exec

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
        """Defines the attributes that will be stored. 

        Any workflow inheriting this class should implement its own version to add their own saving attributes by extending this list
        with ``super()._saving_attrs()``. 
        """
        return ['_random', 'max_round_exec', 'saved_score_limit', 'saved_checkpoint_limit', '_saved_scores']
    
    @property
    def ledger(self) -> Ledger:        
        return self._ledger
    
    @property
    def component(self) -> Component:
        return self._component

    @property
    def dataset(self) -> Dataset:
        return self._dataset
    
    @property
    def n_rounds(self) -> int:
        """Number of rounds have been executed."""
        return self.ledger.n_rounds

    @property
    def isStopped(self) -> bool:
        """Is the workflow stopped. 
        A workflow would stop if the stopping rule suggests so, all documents have been reviewed, or reaches the maximum
        number of rounds (:py:attr:`max_round_exec`)

        The decision is cached so the stopping rule would only be consulted once for efficiency. 
        """
        if self._stopped is None or self._stopped[0] != self.n_rounds:
            # cache the decision given the same round
            self._stopped = (self.n_rounds, 
                             self.component.hasStoppingRule and 
                             self.component.checkStopping(self.ledger.freeze(), workflow=self))
        return self._stopped[1] or self.ledger.isDone or \
               (self.n_rounds > self.max_round_exec and self.max_round_exec > -1)

    @property
    def latest_scores(self) -> np.ndarray:
        """The latest set of document scores."""
        return self._saved_scores[next(reversed(self._saved_scores))]

    def _saveScores(self, scores, overwrite=False):
        """Internal method that workflow should use to save the set of scores to ensure the limit is respected."""
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
        """Saving the workflow state. 

        The workflow state (checkpoint) will be saved a directory containing all essential attributes, including the 
        ledger and the components. 
        The scores will also be stored but if :py:attr:`~saved_score_limit` is not < 0, the scores stored will be 
        incomplete and could only contain the document scores from the latest round. 
        Incomplete scores are sufficient for continuing an incomplete workflow but might not be sufficient to run 
        replay experiments, especially ones that leverage the document scores to estimate the progess 
        (e.g., :py:class:`tarexp.component.stopping.QuantStoppingRule`).

        .. important::
            The underlying dataset is **not** saved along with the workflow as it is supposed to be static
            regardless of containing the gold labels or not. However, dataset hash is stored in the checkpoint to 
            verify the dataset providing at loading time is consistent. 

        Parameters
        ----------
        output_dir
            Path to the directory containing the checkpoints.

        with_component
            Whether the components are saved along with the workflow. Default ``True``.
            If the components are not saved, resuming the workflow would be unsupported.

        overwrite
            Whether overwriting a checkpoint of the current round if exists. 
            Default ``False``.     
        """
        output_dir = Path(output_dir)
        existing_checkpoints = list(output_dir.glob("it_*"))
        if len(existing_checkpoints) >= self.saved_checkpoint_limit:
            for d in sorted(existing_checkpoints, key=lambda d: int(d.name.split("_")[1]))[:-self.saved_checkpoint_limit+1]:
                rmtree(d)

        output_dir = output_dir / f"it_{self.ledger.n_rounds}"
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
        """Load the workflow from checkpoint

        Identical dataset must be provided when loading the workflow. 

        Parameters
        ----------
        saved_path
            Path to either the directory containing the checkpoints or a specific checkpoint. 
            If multiple checkpoints exist in the provided directory, the latest checkpoint will be 
            selected and loaded. 
        
        dataset
            The dataset the workflow was using before saving. 
        
        force
            Whether to skip checking the hash of the dataset. It is not recommanded to turn on since it could 
            result in inconsistent workflow behavior. Default ``False``. 

        """
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
    
    def step(self, force: bool=False):
        """Abstract method defining a TAR step in the workflow. 
        Workflow classes inheriting this meta class should implement this method. 

        Ideally, a TAR step should start from checking whether the workflow has been stopped by consulting
        :py:attr:`~isStopped` and proceed to other parts including retraining the ranker and suggest review documents. 

        Parameters
        ----------
        force 
            Whether to force execute a TAR step if the workflow is already stopped
        """
        raise NotImplementedError
        
    def getMetrics(self, measures: List[OptimisticCost | ir_measures.measures.Measure | str]) -> Dict[MeasureKey, int | float]:
        """Abstract method for providing evaluation metric values at the current round. 

        Each workflow inheriting this meta class can define its own method for evaluating 
        the measures.

        Parameters
        ----------
        measures
            List of evaluation measures. 

        """
        raise NotImplementedError
    
    def makeReplay(self) -> WorkflowReplay:
        """Abstract method for creating replay workflow based on the current TAR workflow. """
        raise NotImplementedError

class WorkflowReplay(Workflow):
    """Meta workflow replay class.

    An existing workflow can be transformed into a :py:class:`~WorkflowReplay` class that freeze the past state of the 
    workflow and replay the process. 
    The replay supports efficient experiments on components and procedures that do not interferes the workflow such as 
    control sets and stopping rules. 
    Since replay operates on an existing workflow, the replay is not savable. 

    Each replay workflow corresponds to an actual workflow. They should be implemented in pair with ``Replay`` at the end
    of the class name for consistence in the module. 

    Parameters
    ----------
    dataset
        Instance of dataset the workflow replay will be operating on. Ideally it should be the same as the one the original
        workflow operated on but not strictly enforced. 
    
    ledger
        The frozen ledger that records the past states of the workflow. 
    
    saved_scores
        The document scores from each TAR round. It is optional depending on the kind of experiments the user intend to run. 
        Experiments such as testing the size or sampling process of the control set do not require the document scores. 

    random_seed
        The random seed for the randomization in the workflow if needed. 

    """
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
        """Replay a TAR step."""
        if self.isStopped:
            return 
        self._replay_round += 1

    def save(self, *args, **kwargs):
        """Not supported in workflow replay."""
        raise NotImplemented("Workflow replay is not savable")
        

class OnePhaseTARWorkflow(Workflow):
    """One Phase TAR Workflow 

    This class defines a one phase TAR workflow that samples a fixed-size set of documents at each round based on certain 
    sampling strategy that takes document scores as input. The suggested documents are reviewed by the human expert (which 
    is simulated through revealing the gold label or other procedure defined in the component). The ranker is then retrained
    based on all the labeled documents and the entire collection is scored and ranked by the updated ranker. 
    Please refer to Yang *et al.* [1]_ for further reference. 

    Please also refer to :py:class:`~Workflow` for other optional parameters. 

    Attributes
    ----------
    review_candidates
        A list of document id that are suggested for review at the current round.

    Parameters
    ----------
    dataset
        Textual dataset the workflow will search on. When running experiments, it should contain gold labels but not strictly 
        required (might raises exceptions depending on the labeler). User could manually provide labels to the workflow in
        each step manually. 

    component
        A combined component that contains ideally all 4 essential roles. Components can be combined by using 
        :py:func:`tarexp.component.base.combine`. 
    
    seed_doc
        A list of documents the human experts reviewed before the iterative process started. Often referred as the 
        **seed set** in eDiscovery community. 

    batch_size
        The number of document the human experts review in each TAR round.

    control_set_size
        Size of the control set. 
    """

    def __init__(self, dataset: Dataset, component: Component, 
                 seed_doc: list = [], batch_size: int = 200,
                 control_set_size: int = 0, 
                 **kwargs):
        super().__init__(dataset=dataset, component=component, **kwargs)
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
                                      self.component.labelDocs(control_set, random=self._random))

        self.review_candidates = seed_doc

    @property
    def _saving_attrs(self):
        return super()._saving_attrs + \
               ['batch_size', 'review_candidates']

    def step(self, force=False):
        """Step of the one phase TAR workflow.

        .. caution::
            If the required roles in the component are missing, the process will be skipped *silently* without
            warnings or excpetions. 
        
        Parameters
        ----------
        force
            Whether to force execute a TAR step if the workflow is already stopped
        """
        if self.isStopped and not force:
            return 

        if self.component.hasLabeler:
            self.ledger.annotate(self.review_candidates, 
                                self.component.labelDocs(self.review_candidates, randome=self._random))

        if self.component.hasRanker:
            self.component.trainRanker(*self.dataset.getTrainingData(self.ledger))
            self._saveScores(
                self.component.scoreDocuments(self.dataset.getAllData())
            )
        
        if self.component.hasRanker and self.component.hasSampler:
            self.review_candidates = self.component.sampleDocs(
                self.batch_size, self.ledger, self.latest_scores
            )

    def getMetrics(self, measures: List[OptimisticCost | ir_measures.measures.Measure | str], 
                   labels: List | np.ndarray = None) -> Dict[MeasureKey, int | float]:
        """Calculate the evaluation measures at the current round
        
        If the underlying dataset does not contain gold labels, they should be provided here in order to calculate the
        values. 
        
        Parameters
        ----------
        measures
            List of evaluation measures. 

        labels
            Labels of the documets in the collection.

        """
        if labels is None:
            if not self.dataset.hasLabels:
                raise ValueError("Labels are not provided.")
            labels = self.dataset.labels
        return evaluate(labels, self.ledger, self.latest_scores, measures)

    def makeReplay(self) -> OnePhaseTARWorkflowReplay:
        """Create a replay workflow that contains records up to the current round. """
        return OnePhaseTARWorkflowReplay(self.dataset, self.ledger.freeze(), self._saved_scores)

# inheritance order matters here -- wants MRO to take replay first
class OnePhaseTARWorkflowReplay(WorkflowReplay, OnePhaseTARWorkflow):
    """Replay workflow for :py:class:`~OnePhaseTARWorkflow`. 

    Parameters
    ----------
    dataset
        Instance of dataset the workflow replay will be operating on. Ideally it should be the same as the one the original
        workflow operated on but not strictly enforced. 
    
    ledger
        The frozen ledger that records the past states of the workflow. 
    
    saved_scores
        The document scores from each TAR round. It is optional depending on the kind of experiments the user intend to run. 
        Experiments such as testing the size or sampling process of the control set do not require the document scores. 
    """

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
        """A list of document id that are suggested for review at the current round."""
        if self.isStopped:
            return self._last_review_candidates
        return self.ledger.getReviewedIds(self._replay_round+1)

    @review_candidates.setter
    def review_candidates(self, candidates):
        self._last_review_candidates = candidates


class TwoPhaseTARWorkflow(OnePhaseTARWorkflow):
    """Two phase workflow extended from :py:class:`~OnePhaseTARWorkflow`. 

    The two phase workflow adds a second phase process that is defined as a ``poststopping`` role in the component, 
    which would be executed after stopping is suggested by the stopping rule. 
    If such role does not exist in the component, an exception would be raised. 

    Please refer to :py:class:`~OnePhaseTARWorkflow` for parameter documentation. 
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.component.checkRole('poststopping'):
            raise ValueError("Two phase workflow requires role `poststopping`.")
        
        self._poststopping_executed = False

    def step(self, *args, **kwargs):
        """Step of the one phase TAR workflow. If stopping is suggested, the second phase review process will be 
        invoked. 
        Please refer to :py:meth:`tarexp.workflow.OnePhaseTARWorkflow.step` for more documentation. 
        """
        super().step(*args, **kwargs)
        if self.isStopped:
            if not self._poststopping_executed:
                self.component.poststopping(
                    self.ledger, self.component, self.dataset
                )
                self._poststopping_executed = True
            else:
                warn("Post stopping role has been executed.")