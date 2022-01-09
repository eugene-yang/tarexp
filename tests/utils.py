import shutil
from pathlib import Path
from functools import lru_cache
import pandas as pd

import tarexp

@lru_cache(maxsize=1)
def getRCV1():
    from sklearn import datasets
    try:
        rcv1 = datasets.fetch_rcv1()
    except:
        shutil.rmtree(Path(datasets.get_data_home()) / "RCV1")
        rcv1 = datasets.fetch_rcv1()
    return tarexp.SparseVectorDataset.from_sparse(rcv1['data']), \
           pd.DataFrame(rcv1['target'].todense().astype(bool), columns=rcv1['target_names'])