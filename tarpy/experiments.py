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

from tarpy.util import iter_with_length, readObj, saveObj, stable_hash

from tarpy.dataset import Dataset, TaskFeeder
from tarpy.component import Component, StoppingRule
from tarpy.workflow import OnePhaseTARWorkflow, Workflow, WorkflowReplay
from tarpy.evaluation import MeasureKey, OptimisticCost


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

def _dispatchRun(setting: dict, 
                output_path: Path, exec_func: function,
                **kwargs):
    plain_setting = _createPlainSetting(setting)
    setting_json = json.dumps(plain_setting)
    setting_str = stable_hash(setting_json)

    if 'dataset' in setting and isinstance(setting['dataset'], Dataset):
        setting_str = setting['dataset'].name + '.' + setting_str

    with (output_path / "task_list.tsv").open('a') as fw:
        fw.write(f"{setting_str}\t{setting_json}\n")

    return plain_setting, exec_func(setting=setting, run_path=output_path / setting_str, **kwargs)

def dispatchEvent(callbacks, event, *args, **kwargs):
    if event in callbacks:
        for f in callbacks[event]:
            f(*args, **kwargs)


@dataclass
class Action:
    should_save: bool = False
    should_stop: bool = False

@dataclass
class Experiment:
    output_path: Path | str
    resume: bool = False
    dump_frequency: int = 10
    random_seed: int = None 
    metrics: List[ir_measures.measures.Measure | OptimisticCost | str] = None
    
    def __post_init__(self):
        if isinstance(self.output_path, str):
            self.output_path = Path(self.output_path)
        if self.metrics is None:
            self.metrics = []
        self._random = np.random.RandomState(self.random_seed)

        self.callbacks = defaultdict(list)

        self._exec_static_kwargs = {
            'resume': self.resume,
            'dump_frequency': self.dump_frequency,
            'metrics': self.metrics,
            'callbacks': self.callbacks
        }
    
    def on(self, event: str, func: callable):
        self.callbacks[event].append(func)
    
    def registerExecArguments(self, kwargs):
        self._exec_static_kwargs.update(kwargs)

    @property
    def available_events(self):
        return ['saved']
    
    @property
    def savable_fields(self):
        return [
            f.name for f in fields(self)
            if not isinstance(getattr(self, f.name), (Dataset, TaskFeeder, iter_with_length))
        ]

    def run(self, disable_tqdm=False, n_processes=1, n_nodes=1, node_id=0):
        if not self.resume and self.output_path.exists():
            raise FileExistsError(f"{self.output_path} already exists.")
        self.output_path.mkdir(parents=True, exist_ok=True)

        setting_iter = self.generateSettings()
        job_iter = ( 
            setting for i, setting in enumerate(setting_iter) 
            if i%n_nodes == node_id
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
                                           **self._exec_static_kwargs)

        if n_processes > 1:
            with Pool(n_processes) as pool:
                ret = list(pool.imap_unordered(
                    dispatcher, it, chunksize=n_processes
                ))
        else:
            ret = list(map(dispatcher, it))
        
        return ret
    
    def generateSettings(self) -> iter_with_length | list:
        raise NotImplementedError

    @staticmethod
    def exec(setting: dict, run_path: Path, **kwargs) -> List[Dict[MeasureKey, int | float]]:
        raise NotImplementedError

@dataclass
class TARExperiment(Experiment):
    tasks: Dataset | List[Dataset] | TaskFeeder = None
    components: Component | List[Component] | iter_with_length = None
    workflow: Workflow = OnePhaseTARWorkflow

    # workflow specific arguments
    batch_size: int | List[int] = 200
    control_set_size: int | List[int] = 0
    workflow_kwargs: dict = None

    repeat_tasks: int = 1
    seed_docs_construction: Tuple[int, int] = (1, 0)
    
    saved_score_limit: int = -1 # experiment setting instead of parameter
    

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
            'workflow_cls': self.workflow
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
                    'saved_score_limit': self.saved_score_limit,
                    'random_seed': self.random_seed,
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

    def _createAction(self, workflow: Workflow):
        return Action(
            should_save = workflow.n_rounds % self.dump_frequency == 0,
            should_stop = workflow.isStopped
        )
        
    @staticmethod
    def exec(setting: dict, run_path: Path, 
             workflow_cls: Workflow, resume: bool, 
             metrics: list, dump_frequency: int, 
             callbacks: dict, **kwargs) -> List[Dict[MeasureKey, int | float]]:
        if run_path.exists():
            if not resume:
                warn(f"Run path {run_path} already exists, task skipped.")
                return 
            
            # resume
            workflow = workflow_cls.load(run_path, setting['dataset'])
            metric_vals: list = readObj(run_path / "exp_metrics.pgz", list)
            if workflow.isStopped:
                return metric_vals
        else:
            workflow = workflow_cls(**setting)
            metric_vals = []

        def _createAction():
            return Action(
                should_save = workflow.n_rounds % dump_frequency == 0,
                should_stop = workflow.isStopped
            )
        
        act = _createAction()
        dispatchEvent(callbacks, "run_begin", act, workflow)
        while True:
            workflow.step()
            act = _createAction()
            dispatchEvent(callbacks, "step_taken", act, workflow)
            metric_vals.append(workflow.getMetrics(metrics))

            if act.should_save or act.should_stop:
                workflow.save(run_path)
                saveObj(metric_vals, run_path / "exp_metrics.pgz")
                dispatchEvent(callbacks, "saved", act, workflow)
            
            if act.should_stop:
                dispatchEvent(callbacks, "stopped", act, workflow)
                break

        act = _createAction()
        dispatchEvent(callbacks, "run_ended", act, workflow)

        return metric_vals


@dataclass
class StoppingExperimentOnReplay(Experiment):
    tasks: Dataset | List[Dataset] | TaskFeeder = None
    replay: WorkflowReplay = None
    saved_exp_path: Path | str = None
    stopping_rules: List[StoppingRule] = None

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
            'stopping_rules': self.stopping_rules
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
            else:
                setting['dataset'] = self.tasks[ setting['dataset'] ]
            setting['save_path'] = self.saved_exp_path / run_hash

            return setting
        
        return iter_with_length(map(mapping, runs), len(runs))
        
    
    def exec(setting: dict, run_path: Path, replay_cls: WorkflowReplay, 
             stopping_rules: List[StoppingRule], dump_frequency: int, 
             callbacks: dict, **kwargs):
        run_path.mkdir(exist_ok=True, parents=True)
        replay = replay_cls.load( setting['save_path'], setting['dataset'] )

        stopping_record = []

        for rule in stopping_rules:
            rule.reset()
        
        dispatchEvent(callbacks, "run_begin", None, replay)
        while True:
            replay.step()
            dispatchEvent(callbacks, "step_taken", None, replay)

            frozen_ledger = replay.ledger
            stopping_record.append({
                MeasureKey(measure=f"{repr(rule)}", target_recall=rule.target_recall): \
                    rule.checkStopping(frozen_ledger)
                for rule in stopping_rules
            })

            if replay.n_rounds % dump_frequency == 0:
                saveObj(stopping_record, run_path / "exp_metrics.pgz")
                dispatchEvent(callbacks, "saved", None, replay)
            
            if replay.isStopped or all(stopping_record[-1].values()):
                # reached the end of replay or all testing stopping rules suggested stopping
                dispatchEvent(callbacks, "stopped", None, replay)
                break

        dispatchEvent(callbacks, "run_ended", None, replay)

        return stopping_record


def createDFfromResults(results, ignore_keys=['seed_doc', 'random_seed', 'saved_score_limit'],
                                 remove_redundant_level=False, 
                                 disable_tqdm=False):
    import pandas as pd

    if isinstance(results, (str, Path)):
        result_path = Path(results)
        if not result_path.is_dir():
            raise NotADirectoryError(f"{result_path} needs to be an experiment directory.")
        if not (result_path / "task_list.tsv").exists():
            raise FileNotFoundError(f"Missing {result_path/'task_list.tsv'}.")
        # load results first
        hashs_settings = [ l.strip().split("\t") for l in (result_path / "task_list.tsv").open() ]
        results = []
        for run_hash, setting in tqdm(hashs_settings, desc='loading', disable=disable_tqdm):
            results.append((json.loads(setting), readObj(result_path / run_hash / "exp_metrics.pgz")))

    ignore_keys = set(ignore_keys)
    index_keys = list(results[0][0].keys() - ignore_keys)
    data = {}
    for r in results:
        index_vals = tuple( r[0][k] for k in index_keys )
        d = pd.DataFrame(r[1])
        d.columns = d.columns.map(
            lambda x: (x.target_recall, x.section, str(x.measure))
        ).rename(['target_recall', 'section', 'measure'])
        data[index_vals] = d
    
    data = pd.concat(data, names=index_keys + ['round'])
    if remove_redundant_level:
        rm_level = []
        for level in data.index.names:
            if data.index.get_level_values(level).unique().size == 1:
                rm_level.append(level)
        data = data.droplevel(rm_level)

    return data.sort_index(axis=0).sort_index(axis=1)