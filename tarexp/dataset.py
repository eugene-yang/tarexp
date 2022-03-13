"""A dataset contains the essential information of the collection for retrieval. 
It is designed to be a static variable throughout the TAR run.

Encoded documents in vectors are required but not limited any form. 
Depending on the intended experiments, the vectors can be generated by scikit-learn TFIDF vectorizer
or even Huggingface Transformers tokenizers. 
We leave this flexibility to the users for further extensions. 

Groud truth labels are essential when running experiments without human intervention. 
It is also used in most evaluation that requires ground truth labels. 
If the workflow is designed to run with actual human reviewing, the labels is no longer required. 
"""

from warnings import warn
from typing import Any, Iterable
from collections import OrderedDict

import numpy as np
import pandas as pd
from scipy.sparse import issparse

from tarexp.ledger import Ledger
from tarexp.util import stable_hash

class Dataset:
    """Meta class of ``TARexp`` dataset. 

    The class defines the basic features for a dataset. 
    All downstream datasets that inherits this class should implement the following properties or method:
    
    * Essentials
    
        * :py:attr:`~identifier`
            The unique identifier of the dataset. It is used in verifying the dataset provided is identical when 
            resuming a workflow. The identifier should summarize both vectors and the labels with a hash that does not 
            depends on the memory location of the variable (e.g. the built-in ``hash()`` function) but the actual 
            content. Utility function :py:func:`tarexp.util.stable_hash` provides such capability. 
    
        * :py:meth:`~ingest`
            It ingests a list of raw text into vectors and stored in the attribute :py:attr:`~_vectors`. 

        * :py:meth:`~getAllData`
            (Optional) It returns all vectors of the documents in the dataset. Ideally, it should returns a copy of the 
            vector but could also be a reference if the colletion is too large to copy in memory. This meta class 
            implemented a simple version but user implementing new dataset class should consider re-implement it to 
            support on-demand processing of the vectors (such as collators in pyTorch). 

        * :py:meth:`~getTrainingData`
            (Optional) It takes a :py:class:`tarexp.ledger.Ledger` as an argument and returns the vectors of reviewed 
            documents and **labels from the ledger**. This meta class also already implemented a simple version but 
            should consider re-implementing for the same reason as :py:meth:`~getAllData`. 
        
        * :py:meth:`~duplicate`
            Returns a copy of the dataset along with any information that should be copied. This method should perform 
            deep copy on all containing objects to prevent memory referencing the prevent fast multi-processing. 

    * Labels (Optional)
    
        * :py:attr:`~labels`
            The labels of the dataset. We recommand implementing this information as a property instead of an attribute 
            of the class to prevent modifying the labels by accident during the workflow. If the labels are intended to 
            be unavailable, please consider raise an ``NotImplemented`` exception instead of ``NotImplementedError`` to 
            reflect the intention. 

        * :py:attr:`~pos_doc_ids` and :py:attr:`~neg_doc_ids`
            The ``set`` of positive and negative docuemnt ids. 

        * :py:meth:`~setLabels`
            The method that returns a **new dataset** that contains the label of all documents in the dataset. It should
            also check all documents are set with a label. Spawning a new instance makes sure that the original dataset 
            instance is not polluted. 
    """
    def __init__(self, name=None):
        self._vectors = None
        self._labels = None
        self._name = name

    @property
    def identifier(self):
        raise NotImplementedError
    
    def __eq__(self, other):
        return self.identifier == other.identifier
    
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
        """Number of documents in the dataset."""
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

    def getAllData(self, copy=False):
        return self._vectors.copy() if copy else self._vectors

    def getTrainingData(self, ledger: Ledger):
        annt = ledger.annotation
        mask = ~np.isnan(annt)
        return self._vectors[mask], annt[mask].astype(bool)
    
    def duplicate(self):
        raise NotImplementedError

    @classmethod
    def from_text(cls, text, **kwargs):
        """Class factory method that returns an instance of the class with ingested text."""
        ds = cls(**kwargs)
        ds.ingest(text)
        return ds


class SparseVectorDataset(Dataset):
    """Dataset with Scipy Sparse Matrix. 

    Parameters
    ----------
    vectorizer
        A function or a class instance that has a ``fit_transform`` method (such as the vectorizers from 
        scikit-learn). 
    """
    
    def __init__(self, vectorizer=None):
        super().__init__()
        self.vectorier = None

        if vectorizer is not None:
            if hasattr(vectorizer, 'fit_transform'):
                self.vectorier = vectorizer.fit_transform
            elif callable(vectorizer):
                self.vectorier = vectorizer
            else:
                raise ValueError(f"Input vectorizer is not a function nore supports `fit_transform`.")
    
    @property
    def _np_data(self):
        return self._vectors.data if issparse(self._vectors) else self._vectors
    
    @property
    def n_docs(self):
        return self._vectors.shape[0]
    
    @property
    def labels(self):
        """Returns a copy of the labels of all docuemnts."""
        if self._labels is None:
            raise AttributeError("Labels are not set for this dataset.")
        return self._labels.copy()

    @property
    def identifier(self):
        if self._labels is None:
            return (self.n_docs, stable_hash(self._np_data), None)    
        return (self.n_docs, stable_hash(self._np_data), stable_hash(self.labels))
    
    @property
    def pos_doc_ids(self) -> set:
        """Returns the ids of the positive documents."""
        if self._labels is None:
            raise AttributeError("Have not set the labels.")
        return set(np.where(self._labels)[0])
    
    @property
    def neg_doc_ids(self) -> set:
        """Returns the ids of the negative documents."""
        if self._labels is None:
            raise AttributeError("Have not set the labels.")
        return set(np.where(~self._labels)[0])
        
    def ingest(self, text, force=False):
        """Ingest the text using the :py:attr:`~vectorizer` and store the vectors in this instance. 
        
        Parameters
        ----------
        text
            A list of text that will be ingested and stored. If the labels are set, the length of the list should be
            identical to the length of labels. 
        
        force
            Whether skipping the test on the length of the text and the labels. 
        """
        if self._labels is not None and len(text) != len(self._labels):
            if not force:
                raise ValueError("Number of input text does not match the labels.")
            self._labels = None
        self._vectors = self.vectorier(text)

    def setLabels(self, labels, inplace=False):
        """Returns a new datset with new labels.

        Parameters
        ----------
        labels
            A list or Numpy array of binary labels. The length should match the number of documents in the dataset.
        
        inplace
            Whether applying this set of labels to the current dataset. If ``True``, the method will replace the labels
            and returns ``None``. Default ``False``. 
        """
        assert len(labels) == self.n_docs
        labels = np.asanyarray(labels).astype(bool)
        if not inplace:
            ret = self.duplicate()
            ret._labels = labels
            return ret
        self._labels = labels
    
    def duplicate(self, deep=False):
        """Duplicate the dataset.

        Parameters
        ----------
        deep
            Whether to perform deep copy on the vectors. Default ``False``. 
        """
        ret = SparseVectorDataset(self.vectorier)
        ret._vectors = self._vectors.copy() if deep else self._vectors
        ret._labels = None if self._labels is None else self._labels.copy()
        return ret
    
    @classmethod
    def from_sparse(cls, matrix):
        """Create a :py:class:`~SparseVectorDataset` instance from a sparse matrix. """
        assert len(matrix.shape) == 2, "Input matrix has to be 2-dimentional."
        ret = cls()
        ret._vectors = matrix
        return ret

class TaskFeeder:
    """Python Iterator that yields review tasks with different set of labels given the same base dataset (a dataset 
    without labels.)

    This class support both iterator in for loop or ``next()`` function and index look up ``[]`` if the list of labels 
    provided has already been materialized (not an iterator). 

    Parameters
    ----------
    dataset
        A :py:class:`~Dataset` instance that does not contain any label. This instance will spawn downstream tasks with
        different labels

    labels
        If a Python dictionary is provided, the key is considered as the names of the tasks and the values are the 
        corresponding labels. The length of all set of labels should be the same as the number of documents provided 
        inbase dataset. 
        
        If a Pandas DataFrame is provided, the columns are considered to be the tasks where the column name are fed
        as the task name. The number of rows in the DataFrame should be the same as the number of documents in the
        base dataset.

        If an iterator is provided, the length of the labels is not checked against the number of documents in the base
        dataset and a warning will be raised. The iterator should yield a tuple of the name of the task and the 
        corresponding labels. The order of should also be stable especially when running experiments across multiple
        machines. If the iterator support length via ``__len__``, the class will respect it.

    """
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
        else:
            raise ValueError(f"Unspported type of labels {labels.__class__}")
    
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
    
    def __contains__(self, item: str):
        if not hasattr(self._raw_labels, '__contains__'):
            raise NotImplemented(f"Provided labels do not support lookup operation.")
        return item in self._raw_labels

    def __getitem__(self, name):
        if not hasattr(self._raw_labels, '__getitem__'):
            raise NotImplemented(f"Provided labels do not support lookup operation.")
        return self._createTask(name, self._raw_labels[name])
    
    @classmethod
    def from_irds(cls, **kwargs):
        """
        .. warning::
            Coming soon.
        """
        raise NotImplementedError