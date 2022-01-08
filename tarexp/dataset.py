from warnings import warn
from typing import Any, Iterable
from collections import OrderedDict

import numpy as np
import pandas as pd
from scipy.sparse import issparse

from tarexp.ledger import Ledger
from tarexp.util import stable_hash

class Dataset:

    def __init__(self, name=None):
        self._vectors = None
        self._labels = None
        self._name = name

    @property
    def identifier(self):
        raise NotImplementedError
    
    def __hash__(self):
        return hash(self.identifier)

    def __repr__(self):
        _name = "" if self._name is None else f": {self._name}"
        return f"<Dataset{_name} ({self.identifier})>"
    
    @property
    def name(self):
        return self._name
    
    @name.setter
    def name(self, name: str):
        if self._name is not None:
            raise AttributeError(f"Name already set.")
        self._name = name

    @property
    def n_docs(self):
        return len(self._vectors)
    
    def __len__(self):
        return self.n_docs

    @property
    def labels(self):
        raise NotImplementedError
    
    @property
    def hasLabels(self):
        return self._labels is not None

    @property
    def pos_doc_ids(self) -> set:
        raise NotImplementedError

    @property
    def neg_doc_ids(self) -> set:
        raise NotImplementedError

    def ingest(self, text, force=False):
        raise NotImplementedError
    
    def setLabels(self, labels, inplace=False):
        raise NotImplementedError

    def getAllData(self):
        raise NotImplementedError

    def getTrainingData(self, ledger: Ledger):
        raise NotImplementedError
    
    def duplicate(self):
        raise NotImplementedError

    @classmethod
    def from_text(cls, text, **kwargs):
        ds = cls(**kwargs)
        ds.ingest(text)
        return ds


class SparseVectorDataset(Dataset):
    
    def __init__(self, vectorizer=None):
        super().__init__()
        if vectorizer is not None and not hasattr(vectorizer, 'fit_transform'):
            raise ValueError(f"Input vectorizer does not support `fit_transform`.")
        self.vectorier = vectorizer
    
    @property
    def _np_data(self):
        return self._vectors.data if issparse(self._vectors) else self._vectors
    
    @property
    def n_docs(self):
        return self._vectors.shape[0]
    
    @property
    def labels(self):
        if self._labels is None:
            raise AttributeError("Labels are not set for this dataset.")
        return self._labels.copy()

    @property
    def identifier(self):
        return (self.n_docs, stable_hash(self._np_data), stable_hash(self.labels))
    
    @property
    def pos_doc_ids(self) -> set:
        if self._labels is None:
            raise AttributeError("Have not set the labels.")
        return set(np.where(self._labels)[0])
    
    @property
    def neg_doc_ids(self) -> set:
        if self._labels is None:
            raise AttributeError("Have not set the labels.")
        return set(np.where(~self._labels)[0])
        
    def ingest(self, text, force=False):
        if self._labels is not None and len(text) != len(self._labels):
            if not force:
                raise ValueError("Number of input text does not match the labels.")
            self._labels = None
        self._vectors = self.vectorier.fit_transform(text)

    def setLabels(self, labels, inplace=False):
        assert len(labels) == self.n_docs
        labels = np.asanyarray(labels).astype(bool)
        if not inplace:
            ret = self.duplicate()
            ret._labels = labels
            return ret
        self._labels = labels
    
    def getAllData(self, copy=False):
        return self._vectors.copy() if copy else self._vectors
    
    def getTrainingData(self, ledger: Ledger):
        annt = ledger.annotation
        mask = ~np.isnan(annt)
        return self._vectors[mask], annt[mask].astype(bool)

    def duplicate(self, deep=False):
        ret = SparseVectorDataset(self.vectorier)
        ret._vectors = self._vectors.copy() if deep else self._vectors
        ret._labels = None if self._labels is None else self._labels.copy()
        return ret
    
    @classmethod
    def from_sparse(cls, matrix):
        assert len(matrix.shape) == 2, "Input matrix has to be 2-dimentional."
        ret = cls()
        ret._vectors = matrix
        return ret

class TaskFeeder:
    def __init__(self, dataset: Dataset, labels: Any):
        assert dataset._labels is None
        self._base_dataset = dataset
        self._raw_labels = labels
        if isinstance(labels, pd.DataFrame):
            assert labels.shape[0] == dataset.n_docs # rows
            self._task_gen = labels.items()
            self._len = labels.shape[1] # columns
        elif isinstance(labels, dict):
            assert all(len(s) == dataset.n_docs for s in labels.values())
            self._task_gen = OrderedDict(sorted(labels.items(), key=lambda x: x[0])).items()
            self._len = len(labels)
        elif isinstance(labels, Iterable):
            warn("Labels provided is an iterable, will trust the elements are "
                 "tuples of name and labels")
            self._task_gen = iter(labels)
            self._len = len(labels) if hasattr(labels, '__len__') else -1
    
    def _createTask(self, name, labels):
        ds = self._base_dataset.setLabels(labels, inplace=False)
        ds.name = name
        return ds

    def __len__(self):
        return self._len
    
    def __iter__(self):
        return self
    
    def __next__(self):
        return self._createTask(*next(self._task_gen))
    
    def __getitem__(self, name):
        if not hasattr(self._raw_labels, '__getitem__'):
            raise NotImplemented(f"Provided labels do not support lookup operation.")
        return self._createTask(name, self._raw_labels[name])
    
    @classmethod
    def from_irds(cls, **kwargs):
        # TODO
        raise NotImplemented