import unittest

import shutil, tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ir_measures import RPrec, P
import tarexp
from tarexp import component
from tarexp.helper import createDFfromResults

from utils import getRCV1

class testWorkflow(unittest.TestCase):

    def setUp(self):
        ds, rel_info = getRCV1()
        self.test_dir = tempfile.mkdtemp()
        self.ds_sample = ds.setLabels(rel_info['C11'])
        setting = component.combine(
                        component.SklearnRanker(LogisticRegression, solver='liblinear'), 
                        component.PerfectLabeler(), 
                        component.RelevanceSampler(), 
                        component.FixedRoundStoppingRule(max_round=20)
                    )()
        seed_doc = np.where(self.ds_sample.labels)[0][0]

        self.workflow = tarexp.OnePhaseTARWorkflow(self.ds_sample, setting, 
                                                   seed_doc=[seed_doc],
                                                   batch_size=200, random_seed=123)
    
    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir)

    def test_saving(self):
        self.workflow.step()
        self.workflow.step()
        self.workflow.step()
        self.workflow.save(self.test_dir)
        self.workflow.step()
        self.workflow.step()
        candidates = self.workflow.review_candidates

        del self.workflow

        workflow_loaded = tarexp.OnePhaseTARWorkflow.load(self.test_dir, self.ds_sample)
        workflow_loaded.step()
        workflow_loaded.step()

        self.assertEqual(workflow_loaded.n_rounds, 4)
        self.assertTrue(all(candidates == workflow_loaded.review_candidates))
    
    def test_replay(self):
        self.workflow.step()
        self.workflow.step()
        candidates = set(self.workflow.review_candidates)
        self.workflow.step()

        replay = self.workflow.makeReplay()
        
        replay.step()
        self.assertTrue( (replay.latest_scores == self.workflow._saved_scores[0]).all() )
        replay.step()
        self.assertTrue( (replay.latest_scores == self.workflow._saved_scores[1]).all() )
        self.assertEqual( len(set(replay.review_candidates) - candidates), 0)


class testExperiments(unittest.TestCase):

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.ds, self.rel_info = getRCV1()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_exp_running(self):

        settings = [
            component.combine(component.SklearnRanker(LogisticRegression, solver='liblinear'), 
                              component.PerfectLabeler(), component.RelevanceSampler(), 
                              component.FixedRoundStoppingRule(max_round=2))(),
            component.combine(component.SklearnRanker(LogisticRegression, solver='liblinear'), 
                              component.PerfectLabeler(), component.UncertaintySampler(), 
                              component.FixedRoundStoppingRule(max_round=2))()
        ]

        exp = tarexp.TARExperiment(
            self.test_dir/"org", random_seed=123,
            metrics=[
                RPrec, P@10, 
                tarexp.OptimisticCost(target_recall=0.8, cost_structure=(1,10,1,10))
            ],
            tasks=tarexp.TaskFeeder(self.ds, self.rel_info[['C11', 'GPRO']]),
            components=settings,
            workflow=tarexp.OnePhaseTARWorkflow, batch_size=[200, 100]
        )

        results = exp.run(n_processes=1, resume=False, dump_frequency=2)

        self.assertEqual(len(results), 8)

        self.assertTrue(all(
            createDFfromResults(results, remove_redundant_level=True) == \
            createDFfromResults(self.test_dir/"org", remove_redundant_level=True)
        ))

        replay_exp = tarexp.StoppingExperimentOnReplay(
            self.test_dir/"replay", random_seed=123,
            tasks=tarexp.TaskFeeder(self.ds, self.rel_info[['C11', 'GPRO']]),
            replay=tarexp.OnePhaseTARWorkflowReplay,
            saved_exp_path=self.test_dir/"org",
            stopping_rules=[
                component.FixedRoundStoppingRule(max_round=0),
                component.FixedRoundStoppingRule(max_round=1),
                component.NullStoppingRule()
            ]
        )

        results = replay_exp.run(resume=False, dump_frequency=1)

        self.assertEqual(len(results), 8)
