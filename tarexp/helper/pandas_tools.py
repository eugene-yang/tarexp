import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
import json

from tarexp.util import readObj

def _createDFfromWorkflowResult(result):
    d = pd.DataFrame(result)
    d.columns = d.columns.map(
        lambda x: (x.target_recall, x.section, str(x.measure))
    ).rename(['target_recall', 'section', 'measure'])
    return d

def createDFfromResults(results, ignore_keys=['seed_doc', 'random_seed', 'saved_score_limit'],
                                 remove_redundant_level=False, 
                                 disable_tqdm=False):
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

    if isinstance(results[0], dict):
        # only one run
        return _createDFfromWorkflowResult(results)

    ignore_keys = set(ignore_keys)
    index_keys = list(results[0][0].keys() - ignore_keys)
    data = {}
    for r in results:
        index_vals = tuple( r[0][k] for k in index_keys )
        data[index_vals] = _createDFfromWorkflowResult(r[1])
    
    data = pd.concat(data, names=index_keys + ['round'])
    if remove_redundant_level:
        rm_level = []
        for level in data.index.names:
            if data.index.get_level_values(level).unique().size == 1:
                rm_level.append(level)
        data = data.droplevel(rm_level)

    return data.sort_index(axis=0).sort_index(axis=1)
    