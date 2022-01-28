from dataclasses import FrozenInstanceError
import unittest

import numpy as np
import pandas as pd

import tarexp
from utils import getRCV1

class testDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.ds, self.rel_info = getRCV1()
    
    def test_sparse_dataset(self):
        ds = self.ds.setLabels(self.rel_info['CCAT'])
        self.assertTrue((ds.labels == self.rel_info['CCAT']).all())
        self.assertEqual(ds.n_docs, 804414)
        pos_ids = ds.pos_doc_ids
        self.assertEqual(len(pos_ids), 381327)
        self.assertTrue(np.isin([1, 373, 627, 804140], list(pos_ids)).all())
        self.assertEqual(ds.n_docs - len(pos_ids), len(ds.neg_doc_ids))

    def test_taskfeeder(self):
        feeder = tarexp.TaskFeeder(self.ds, self.rel_info)
        self.assertEqual(len(feeder), self.rel_info.shape[1])
        
        self.assertEqual(feeder['CCAT'], self.ds.setLabels(self.rel_info['CCAT']))

        ds_sample = next(feeder)
        self.assertTrue((ds_sample.labels == self.rel_info[ds_sample.name]).all())

class testLedger(unittest.TestCase):
    
    def test_annotate(self):
        ledger = tarexp.Ledger(100)
        with self.assertRaises(AssertionError):
            ledger.annotate([0, 1], [True])

        ledger.annotate(np.arange(0, 100, 10), [True]*10)

        with self.assertRaises(AssertionError):
            ledger.annotate(np.arange(0, 100, 3), [False]*34)
        
        ledger.annotate(np.arange(1, 100, 11), [False]*9)

        self.assertEqual(ledger.n_rounds, 1)
        self.assertEqual(ledger.n_pos_annotated, 10)
        self.assertEqual(ledger.n_neg_annotated, 9)

        frozen = ledger.freeze()

        with self.assertRaises(FrozenInstanceError):
            frozen.annotate([7], [False])
        
        self.assertTrue((frozen.annotation == ledger.annotation)[frozen.annotated].all())

        frozen_zero = ledger.freeze_at(0)
        self.assertEqual(frozen_zero.n_pos_annotated, 10)
        self.assertEqual(frozen_zero.n_neg_annotated, 0)

        self.assertTrue((frozen.freeze_at(0).annotation == frozen_zero.annotation)[frozen_zero.annotated].all())
        

