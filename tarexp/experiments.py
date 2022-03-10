"""TAR inherits both the large topic-to-topic variability of IR tasks, and the strong dependence on initial conditions 
and random seeds of active learning processes. Multiple collections, topics, and runs are necessary to reliably demonstrate 
that one approach dominates another.

We support spawning multiple processes both across machines on a network, and in multiple threads on appropriate hardware.  
The method ``.run`` dispatches the TAR tasks with runtime settings. In the above example, experiments will run on the first 
of the two machines with two processes on each, resulting in all four tasks being run simultaneously. The ``.run`` method 
returns the per-round metric values of all the experiment tasks running on the node.  
"""

from __future__ import annotations
from dataclasses import dataclass, fields
from functools import partial
from collections import defaultdict
from pathlib import Path
from multiprocessing import Pool
from typing import Any, Dict, Iterable, List, Tuple
from itertools import product

import json
from warnings import warn

import numpy as np
from tqdm.auto import tqdm
import ir_measures

from tarexp.util import iter_with_length, readObj, saveObj, stable_hash

from tarexp.dataset import Dataset, TaskFeeder
from tarexp.component import Component, StoppingRule
from tarexp.workflow import OnePhaseTARWorkflow, Workflow, WorkflowReplay
from tarexp.evaluation import MeasureKey, OptimisticCost


def _make_list(it):
    return [it] if not isinstance(it, Iterable) else it

def _generateProducts(things: Dict[List[Any]]):
    if len(things) == 0:
        return [{}]
    keys, values = zip(*things.items())
    return iter_with_length(
        ( dict(zip(keys, vs)) for vs in product(values) ),
        length=int(np.prod([ len(v) for v in values]))
    )

def _toRepr(e):
    if isinstance(e, (int, float)):
        return e
    if isinstance(e, tuple):
        return tuple( _toRepr(i) for i in e )
    if hasattr(e, 'name'):
        return e.name
    if hasattr(e, '__name__'):
        return e.__name__
    return repr(e)

def _createPlainSetting(setting: dict | list) -> dict | list:
    if isinstance(setting, dict):
        return {
            k: _createPlainSetting(v) if isinstance(v, (list, dict)) else _toRepr(v)
            for k, v in setting.items()
        }
    else:
        return [
            _createPlainSetting(v) if isinstance(v, (list, dict)) else _toRepr(v)
            for v in setting
        ]

def _dispatchRun(setting: dict, output_path: Path, exec_func: function, **kwargs):
    plain_setting = _createPlainSetting(setting)
    setting_json = json.dumps(plain_setting)
    setting_str = stable_hash(setting_json)

    if 'dataset' in setting and isinstance(setting['dataset'], Dataset):
        setting_str = setting['dataset'].name + '.' + setting_str

    with (output_path / "task_list.tsv").open('a') as fw:
        fw.write(f"{setting_str}\t{setting_json}\n")

    return plain_setting, exec_func(exp_setting=setting, run_path=output_path / setting_str, **kwargs)

def _dispatchEvent(callbacks, event, *args, **kwargs):
    if event in callbacks:
        for f in callbacks[event]:
            f(*args, **kwargs)


@dataclass
class Action:
    """Action controler for feedback functions in experiments.

    Feedback functions recieve an :py:class:`~Action` class that each attribute modifiable for suggesting
    the experiment running function to perform certain operation. 

    """
    should_save: bool = False
    """Suggest the workflow to save a checkpoint at this round."""

    should_stop: bool = False
    """Suggest the workflow to stop after this round."""

@dataclass
class Experiment:
    """:py:class:`~Experiment` is implemented as a Python Dataclass, which contains information for experiments but not 
    the states, which are determined by the checkpoint of each individual run on disk. 
    The attributes (or the properties of the class) are designed to be static configurations of the experiments. 
    Runtime configurations are passed as arguments to the :py:meth:`~run` method.

    All experiments inheriting this class should implement the static method :py:meth:`~exec` which describes how
    a run should execute and :py:meth:`~generateSettings` which generates a list of experimenting settings that will be 
    yielded as TAR runs. The configuration would be passed as the first arugment (or keyed as ``setting``) to this 
    static method. 

    All downstream classes should also register any additional arguments through :py:meth:`~registerExecArguments` to 
    ensure these configurations are passed into the :py:meth:`~exec` static method.
    """
    output_path: Path | str
    """Output directory of the experiment."""
    
    random_seed: int = None 
    """Random seed used in the experiment and all experiment runs."""
    
    metrics: List[ir_measures.measures.Measure | OptimisticCost | str] = None
    """Evaluation metrics that would be calculated and stored at each round of the runs."""

    saved_score_limit: int = -1
    """Number of rounds the document scores each workflow would be storing. Negative indicates no limitation."""
    
    saved_checkpoint_limit: int = 2
    """Maximum number of checkpoints would be stored on disk. Older checkpoints would be deleted silently."""
    
    max_round_exec: int = -1
    """Maximum number of rounds each run would be allowed to execute."""
    
    def __post_init__(self):
        if isinstance(self.output_path, str):
            self.output_path = Path(self.output_path)
        if self.metrics is None:
            self.metrics = []
        self._random = np.random.RandomState(self.random_seed)

        self.callbacks = defaultdict(list)

        self._exec_static_kwargs = {
            'saved_score_limit': self.saved_score_limit,
            'saved_checkpoint_limit': self.saved_checkpoint_limit,
            'max_round_exec': self.max_round_exec,
            'metrics': self.metrics,
            'random_seed': self.random_seed,
            'callbacks': self.callbacks,
        }
    
    def on(self, event: str, func: callable):
        """Register a callback function tied to a certain event. 
        List of events available for each experiment is stored in :py:attr:`~available_events`.

        Parameters
        ----------
        event
            Name of of the event the callback function would be invoked. 
        
        func
            Callback funtion that will take in an :py:class:`~Action` instance and the :py:class:`tarexp.workflow.Workflow`
            instance as arguments. 
        """
        self.callbacks[event].append(func)
    
    def registerExecArguments(self, kwargs: Dict[str, Any]):
        """Register static experiment arguments.

        Parameters
        ----------
        kwargs
            Dictionary of arguments where the keys and values are the name and values of the argument.
        """
        self._exec_static_kwargs.update(kwargs)

    @property
    def available_events(self):
        """List of available events of this experiment."""
        return ['saved']
    
    @property
    def savable_fields(self):
        """List of attributes that would be saved in the experiment directory."""
        return [
            f.name for f in fields(self)
            if not isinstance(getattr(self, f.name), (Dataset, TaskFeeder, iter_with_length))
        ]

    def run(self, disable_tqdm=False, n_processes=1, n_nodes=1, node_id=0, 
            resume=False, dump_frequency=10, **runtime_kwargs):
        """Execute all experiment runs.

        This method natively support multiprocessing on a single machine and parallel processing on multiple machines.
        However, when running on multiple machines, user needs to manually invoke this method with proper ``n_nodes`` and
        ``node_id`` values in order to let each machine execute their own parts of experiments.

        Arguments of this method are designed to be runtime configurations, including ones that will eventually be passed
        to the underlying workflow instances.  

        Parameters
        ----------
        disable_tqdm
            Whether to show the progress bar. Default is ``False`` (showing the bar). 
        
        n_process
            Number of processers to use on this machine. 
        
        n_nodes
            Number of machines running this set of experiments in total. 

        node_id
            The index of this machine among all machines starting from ``0``. This value should be < ``n_nodes``. 
        
        resume
            Whether to resume from existing runs. Default is ``True``. This value will be passed to the workflow and 
            expect the workflow to respect it. 
        
        dump_frequency
            The frequency of the workflow to save a checkpoint on disk. 
        
        **runtime-kwargs
            Other runtime arguments to pass to the workflow instance. 
        """
        if not resume and self.output_path.exists():
            raise FileExistsError(f"{self.output_path} already exists.")
        self.output_path.mkdir(parents=True, exist_ok=True)

        setting_iter = self.generateSettings()
        job_iter = ( 
            setting for i, setting in enumerate(setting_iter) 
            if i%n_nodes == node_id
            if setting is not None
        )
        n_jobs_local = len(setting_iter)//n_nodes + int((len(setting_iter))%n_nodes > node_id)

        if node_id == 0:
            exp_setting_dict = {
                **{ f: getattr(self, f) for f in self.savable_fields },
                "n_jobs": len(setting_iter)
            }
            # NOTE: probably want to check whether it is the same if exists
            saveObj(exp_setting_dict, self.output_path / "exp_setting.pgz")

        it = tqdm(job_iter, desc=f"Total tasks on node {node_id}", 
                  disable=disable_tqdm, total=n_jobs_local)
        
        dispatcher = partial(_dispatchRun, output_path=self.output_path, 
                                           exec_func=self.__class__.exec,
                                           **self._exec_static_kwargs,
                                           resume=resume, 
                                           dump_frequency=dump_frequency, 
                                           **runtime_kwargs)

        if n_processes > 1:
            with Pool(n_processes) as pool:
                ret = list(pool.imap_unordered(
                    dispatcher, it, chunksize=1
                ))
        else:
            ret = list(map(dispatcher, it))
        
        return ret
    
    def generateSettings(self) -> iter_with_length | list:
        """Generate a list experimenting settings.

        Any experiment class that inherits this class should implement its own version since each experiment should define
        its own set of attributes that could be experimenting with (or more complex ways to generate sensible combinations). 
        The rest of the attributes should be static. 
        """
        raise NotImplementedError

    @staticmethod
    def exec(setting: dict, run_path: Path, **kwargs) -> List[Dict[MeasureKey, int | float]]:
        """Executing a TAR run for experiment.

        Any experiment class that inherits this class should implement its own version of this method. 
        This method describe how an experiment of the workflow will be executed, including when the metrics are calcuated
        and when the callback functions are invoked. 

        .. note::
            This method can be confused with the :py:meth:`tarexp.workflow.Workflow.step` method. However, ``exec`` 
            focuses on the experimenting side which should not modify the behavior of a workflow. 
        """
        raise NotImplementedError

@dataclass
class TARExperiment(Experiment):
    """An experiment that executes a TAR workflow. 
    
    This experiment class is designed to be compatible with any workflow that inherits :py:class:`tarexp.workflow.Workflow`
    class. 

    Please refer to :py:class:`~Experiment` for properties inherited from it.

    .. important::
        Any attribute that marked as **experimentable** will contribute to the process of generating the combination of the 
        experiment runs. For example, if 2 tasks, 3 sets of components, 3 batch sizes, and 10 control set sizes are provided,
        :py:meth:`~generateSettings` will yield 2x3x3x10 = 180 runs. 
    """
    tasks: Dataset | List[Dataset] | TaskFeeder = None
    """**Experimentable** (A list of) dataset instance or :py:class:`tarexp.dataset.TaskFeeder` class.
    Each task is considered as an independent TAR review project which consists of a collection and a set of gold labels. 
    """
    
    components: Component | List[Component] | iter_with_length = None
    """**Experimentable** (A list of) combined components for experiments."""
    
    workflow: Workflow = OnePhaseTARWorkflow
    """Workflow that will be used for experiment."""

    # workflow specific arguments
    batch_size: int | List[int] = 200
    """**Experimentable** (A list of) batch sizes for the TAR workflow."""
    
    control_set_size: int | List[int] = 0
    """**Experimentable** (A list of) control set sizes for TAR workflow."""
    
    workflow_kwargs: dict = None
    """**Experimentable** Other experimenting arguments. 
    If the value of the pair in the dictionary is a list, it will 
    be considered as an experimentable arguments and also contributes to the combination.
    """

    
    repeat_tasks: int = 1
    """**Experimentable** Number of times each task will be replicated with different random seed."""
    
    seed_docs_construction: Tuple[int, int] = (1, 0)
    """Number of positive and negative seed documents will be generated for each run."""

    def __post_init__(self):
        super().__post_init__()
        
        if isinstance(self.tasks, Dataset):
            self.tasks = [self.tasks]
        if isinstance(self.components, Component):
            self.components = [self.components]
        
        self.batch_size = _make_list(self.batch_size)
        self.control_set_size = _make_list(self.control_set_size)
        
        if self.workflow_kwargs is not None:
            self.workflow_kwargs = {
                k: v if isinstance(v, Iterable) else [v]
                for k, v in self.workflow_kwargs.items()
            }
        else:
            self.workflow_kwargs = {}
        
        self.registerExecArguments({
            'workflow_cls': self.workflow,
        })
    
    @property
    def available_events(self):
        return super().available_events + \
            ['run_begin', 'step_taken', 'stopped', 'run_ended']
    
    def _createSeed(self, dataset: Dataset):
        pos_docs = self._random.permutation(list(dataset.pos_doc_ids))
        neg_docs = self._random.permutation(list(dataset.neg_doc_ids))
        npos, nneg = self.seed_docs_construction
        return [
            [ *pos_docs[i*npos: (i+1)*npos], *neg_docs[i*nneg: (i+1)*nneg] ]
            for i in range(self.repeat_tasks)
        ]

    def generateSettings(self):
        """Generate an iterator that yields experiment settings in dictionaries. 

        In this experiments, :py:attr:`~tasks`, :py:attr:`~components`, :py:attr:`~repeat_tasks`, :py:attr:`~batch_size`,
        :py:attr:`~control_set_size`, and any other values in :py:attr:`~workflow_kwargs` will be contributed to the 
        combination.
        """
        other_settings_prod = _generateProducts(self.workflow_kwargs)
        nruns = len(self.tasks) * len(self.components) * self.repeat_tasks * \
                len(self.batch_size) * len(self.control_set_size) * \
                len(other_settings_prod)
                
        tasks = enumerate(self.tasks) if not isinstance(self.tasks, TaskFeeder) else self.tasks

        return iter_with_length(
            (
                {
                    'dataset': ds, 
                    'component': comp,
                    'seed_doc': seed,
                    'irep': irep,
                    'batch_size': bs,
                    'control_set_size': ctl_set,
                    'random_seed': self.random_seed + irep,
                    **other_settings
                }
                for ds in tasks
                for comp in self.components
                for irep, seed in enumerate(self._createSeed(ds))
                for bs in self.batch_size
                for ctl_set in self.control_set_size
                for other_settings in other_settings_prod
            ),
            length = nruns
        )
        
    @staticmethod
    def exec(exp_setting: dict, run_path: Path, 
             workflow_cls: Workflow, resume: bool, 
             metrics: list, dump_frequency: int, 
             callbacks: dict, **static_setting) -> List[Dict[MeasureKey, int | float]]:
        """Execute a run of experiment using the workflow specified in :py:attr:`~workflow`.
        This method is used internally by the dispatcher invoked by :py:meth:`tarexp.experiments.Experiment.run`.
        """

        if run_path.exists():
            if not resume:
                warn(f"Run path {run_path} already exists, task skipped.")
                return 
            
            # resume
            workflow = workflow_cls.load(run_path, exp_setting['dataset'])
            metric_vals: list = readObj(run_path / "exp_metrics.pgz", list)
            if workflow.isStopped:
                return metric_vals
        else:
            del static_setting['random_seed'] # it is also an experimenting variable here
            workflow = workflow_cls(**exp_setting, **static_setting)
            metric_vals = []

        def _createAction():
            return Action(
                should_save = workflow.n_rounds % dump_frequency == 0,
                should_stop = workflow.isStopped
            )
        
        act = _createAction()
        _dispatchEvent(callbacks, "run_begin", act, workflow)
        while True:
            workflow.step()
            act = _createAction()
            _dispatchEvent(callbacks, "step_taken", act, workflow)
            metric_vals.append(workflow.getMetrics(metrics))

            if act.should_save or act.should_stop:
                workflow.save(run_path)
                saveObj(metric_vals, run_path / "exp_metrics.pgz")
                _dispatchEvent(callbacks, "saved", act, workflow)
            
            if act.should_stop:
                _dispatchEvent(callbacks, "stopped", act, workflow)
                break

        act = _createAction()
        _dispatchEvent(callbacks, "run_ended", act, workflow)

        return metric_vals


@dataclass
class StoppingExperimentOnReplay(Experiment):
    """Stopping Rule experiments on existing TAR runs using :py:class:`tarexp.workflow.WorkflowReplay`.
    
    This experiment invokes different stopping rules at each replay round to test whether the rule suggests stopping. 
    All runs in the provided TAR experiment directory will be tested. 

    .. important::
        Since stopping rules are only tested on the replays, any rule that intervenes with the TAR process (such as 
        changing the sampling of the documents for estimating progress) cannot be tested with this experiment. User 
        should execute individual TAR runs by using :py:class:`~TARExperiment` to test those stopping rules. 
    """
    saved_exp_path: Path | str = None
    """Path to the directory of the TAR runs."""
    
    tasks: Dataset | List[Dataset] | TaskFeeder = None
    """List of dataset instance or TaskFeeder that provides the datasets for replay experiments."""
    
    replay: WorkflowReplay = None
    """The replay workflow class that corresponds to the workflow used to generate the TAR runs provided in 
    :py:attr:`~saved_exp_path`.
    """
    
    stopping_rules: List[StoppingRule] = None
    """List of stopping rules that will be tested on the replay workflow."""

    exp_early_stopping: bool = True
    """Whether to early stop the replay if all stopping rule tested have already suggested stopping."""

    def __post_init__(self):
        super().__post_init__()

        if isinstance(self.saved_exp_path, str):
            self.saved_exp_path = Path(self.saved_exp_path)
        
        if not self.saved_exp_path.is_dir() or \
           not (self.saved_exp_path / "exp_setting.pgz") or \
           not (self.saved_exp_path / "task_list.tsv"):
            raise OSError(f"{self.saved_exp_path} is not a valid experiment directory.")

        if isinstance(self.tasks, list):
            self.tasks = { t.name: t for t in self.tasks }

        self.registerExecArguments({
            'replay_cls': self.replay,
            'stopping_rules': self.stopping_rules,
            'exp_early_stopping': self.exp_early_stopping
        })

    @property
    def available_events(self):
        return super().available_events + \
            ['run_begin', 'step_taken', 'stopped', 'run_ended']

    def generateSettings(self):
        runs = list((self.saved_exp_path / "task_list.tsv").open())
        def mapping(line):
            run_hash, setting = line.strip().split("\t")
            setting = json.loads(setting)
            if isinstance(self.tasks, Dataset):
                setting['dataset'] = self.tasks
            elif setting['dataset'] in self.tasks:
                setting['dataset'] = self.tasks[ setting['dataset'] ]
            else:
                warn(f"Dataset `{setting['dataset']}` is not provided, skipped replay.")
                return None

            setting['save_path'] = self.saved_exp_path / run_hash

            return setting
        
        return iter_with_length(map(mapping, runs), len(runs))
        
    
    def exec(exp_setting: dict, run_path: Path, replay_cls: WorkflowReplay, 
             stopping_rules: List[StoppingRule], metrics: list, 
             dump_frequency: int, exp_early_stopping: bool, callbacks: dict, **kwargs):
        """Execute a run of experiment using the replay workflow specified in :py:attr:`~replay_workflow`.
        This method is used internally by the dispatcher invoked by :py:meth:`tarexp.experiments.Experiment.run`.

        .. note::
            Since the stopping rule runs are designed to be fast, resume is not supported in this experiment.
        """
        run_path.mkdir(exist_ok=True, parents=True)
        replay = replay_cls.load( exp_setting['save_path'], exp_setting['dataset'] )

        stopping_record = []

        for rule in stopping_rules:
            rule.reset()
        
        _dispatchEvent(callbacks, "run_begin", None, replay)
        while True:
            replay.step()
            _dispatchEvent(callbacks, "step_taken", None, replay)

            frozen_ledger = replay.ledger
            stopping_record.append({
                **{
                    MeasureKey(measure=f"{repr(rule)}", target_recall=rule.target_recall): \
                        rule.checkStopping(frozen_ledger, workflow=replay)
                    for rule in stopping_rules
                },
                **replay.getMetrics(metrics)
            })

            if replay.n_rounds % dump_frequency == 0:
                saveObj(stopping_record, run_path / "exp_metrics.pgz")
                _dispatchEvent(callbacks, "saved", None, replay)
            
            if replay.isStopped or (exp_early_stopping and all(stopping_record[-1].values())):
                # reached the end of replay or all testing stopping rules suggested stopping
                _dispatchEvent(callbacks, "stopped", None, replay)
                break

        _dispatchEvent(callbacks, "run_ended", None, replay)

        return stopping_record