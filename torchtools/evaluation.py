# AUTOGENERATED! DO NOT EDIT! File to edit: 300_evaluation.ipynb (unless otherwise specified).

__all__ = ['augment_params', 'fitting_params', 'metric_cols', 'metric_cols_classification', 'loss_cols', 'fn_cols',
           'experiment_cols', 'var_cols', 'create_target_col', 'create_ds_col', 'create_col_config_col',
           'pre_process_results', 'EvalConfig', 'get_base_dir', 'yval_for_classification', 'basic_eval', 'filt_df',
           'get_top', 'get_best_params', 'rough_param_comparison', 'get_best_params']

# Cell
from .core import *
from .data import *
from .models import *
from .datasets import *
from .augmentations import *
#from torchtools.datablock import *
from .dataloader import *
from .experiments import *
from .configs import *

# Cell
import pandas as pd
import numpy as np
from fastai.basics import *
from datetime import datetime, timedelta

import re

# Cell
import pandas as pd
import numpy as np
from fastai.basics import *
from datetime import datetime, timedelta

import re

# Cell
#constants
augment_params = ['N', 'magnitude']
fitting_params = ['n_epochs', 'max_lr', 'wd', 'pct_start', 'div_factor']
#     metric_cols = [c for c in df.columns if 'metric_' in c and '_value' in c]
metric_cols = ['unweighted_profit_0_value', 'unweighted_profit_05_1_value']
metric_cols_classification = ['accuracy_0_value']

loss_cols = ['val_loss', 'trn_loss']
fn_cols=['val_preds', 'test_preds']#, 'model_fn']
experiment_cols = ['arch', 'bs', 'ds_id']
var_cols = ['Timestamp']

# Cell
def create_target_col(df_results):
    pat = r'_((1yml)|(1yhc)|(2y)|(1yclass))'
    ds_ids = df_results.ds_id.values
    ds_id_matches=[re.search(pat, d) for d in ds_ids]
    df_results['target'] = [m.group(1) if m is not None else 'undefined' for m in ds_id_matches]

def create_ds_col(df_results):
    pat = r'^((bi_sample_anon)|(bi_sample_pruned_anon)|(ts_experiments_anon_ts_exp_2020818))'
    ds_ids = df_results.ds_id.values
    ds_id_matches=[re.search(pat, d) for d in ds_ids]
    df_results['ds'] = [m.group(1) if m is not None else 'undefined' for m in ds_id_matches]

def create_col_config_col(df_results):
    pat = r'((\d+sl).*)_((1yml)|(1yhc)|(2y)|(1yclass))'
    ds_ids = df_results.ds_id.values
    ds_id_matches=[re.search(pat, d) for d in ds_ids]
    df_results['col_config'] = [m.group(1) if m is not None else 'undefined' for m in ds_id_matches]

def pre_process_results(df_results):
    create_target_col(df_results)
    create_ds_col(df_results)
    create_col_config_col(df_results)
    df_results['Timestamp'] = pd.to_datetime(df_results['Timestamp'])

# Cell
def _get_base_dir(is_colab=False):
    return Path('./') if not is_colab else Path('~/google-drive').expanduser()

def _results_fn(is_colab=False, is_class=False):
    if is_colab:
        results_fn='results_colab.csv' if not is_class else 'results_colab_class.csv'
    else:
        results_fn='results_exploration.csv' if not is_class else 'results_exploration_class.csv'
    return results_fn

class EvalConfig:
    '''
    eval configuration wrapper
    #export
    #constants
    #RESULTS_DIR='experiments/results'
    #PREDS_DIR='experiments/preds'


    '''
    def __init__(self, is_colab, is_class, df_base_path, results_loc='experiments/results',
                preds_loc='experiments/preds', preprocess=True):
        self.is_colab, self.is_class, self.df_base_path = is_colab, is_class, df_base_path
        self.results_loc, self.preds_loc = results_loc, preds_loc

    @property
    def base_dir(self): return _get_base_dir(self.is_colab)
    @property
    def preds_dir(self): return self.base_dir/self.preds_loc
    @property
    def results_dir(self): return self.base_dir/self.results_loc
    @property
    def df_results_path(self): return self.results_dir/_results_fn(self.is_colab, self.is_class)
    @property
    def df_results(self):
        if not hasattr(self, '_df_results'):
            self._df_results = pd.read_csv(self.df_results_path)
            pre_process_results(self._df_results)
        return self._df_results

    @property
    def df_base(self):
        if not hasattr(self, '_df_base'): self._df_base = pd.read_csv(self.df_base_path)
        return self._df_base
    @property
    def m_cols(self):
        return metric_cols_classification if self.is_class else metric_cols




# Cell
def get_base_dir(is_colab=False):
    return Path('./') if not is_colab else Path('~/google-drive').expanduser()

# Cell
def _reload_preds(eval_conf, idx, test=False):
    if not test:
        fn = eval_conf.preds_dir/eval_conf.df_results.iloc[idx]['val_preds']
    else:
        fn=eval_conf.preds_dir/eval_conf.df_results.iloc[idx]['test_preds']
    return torch.load(fn)

#export
def _average_preds(eval_conf, idxs, test=False, normalize=False):
    preds = np.array([_reload_preds(eval_conf, idx, test).numpy() for idx in idxs])
    if normalize:
        mean,std = preds.mean(), preds.std()
        preds = (preds-mean)/std
    return tensor(preds.mean(0))

#export
def yval_for_classification(df, splits):
    n_train = len(splits[0])
    print(n_train)
    y, maps = cats_from_df(df, ['y2'], n_train, add_na=False)
    print(maps)
    return tensor(y[splits[1]])

# Cell
def _get_bet_idxs(preds, threshold=None, quantile=0.9):
    if threshold is None: threshold = np.quantile(preds, quantile)
    return torch.where(preds>threshold)[0]


def basic_eval(eval_conf, model_idx, test=False, threshold=None, quantile=0.9):
    if is_listy(model_idx):
        preds = _average_preds(eval_conf, model_idx, test=test)
    else:
        preds = _reload_preds(eval_conf, model_idx, test=test)

    if test:
        splits=[L(range(160000)), L(range(185000,210000))] ##TODO put into eval_conf
    else:
        splits=[L(range(160000)), L(range(160000,185000))]

    bet_idxs = _get_bet_idxs(preds, threshold, quantile)
    print(eval_conf.df_base.iloc[splits[1]].iloc[bet_idxs][['y0', 'y1']].agg(['mean', 'sum', 'count']))
    if is_listy(model_idx):
        preds_all = [_reload_preds(eval_conf, idx, test=test) for idx in model_idx]
        bet_idxs_all = [_get_bet_idxs(preds, threshold, quantile) for preds in preds_all]
        res_all = [eval_conf.df_base.iloc[splits[1]].iloc[bet_idxs][['y1']].mean().values for bet_idxs in bet_idxs_all]
        print(f'single results {res_all}')
        print(f'mean result {np.array(res_all).mean()}')

# Cell
def _extract_date(d, key):
    return d.pop(key, None)

def filt_df(df_results, filt):
    '''
    filter results dataframe according to `filt`
    filt: dictionary with respective columns as keys and a list of values to be filtered fo
    '''
    filt = filt.copy()
    start_date = _extract_date(filt, 'start_date')
    end_date = _extract_date(filt, 'end_date')

    crits = [df_results[k].isin(listify(filt[k])) for k in filt.keys()]
    crit_filt = np.all(crits, axis=0)
    if start_date is not None: crit_filt = np.logical_and(crit_filt, (df_results['Timestamp']>=start_date).values)
    if end_date is not None: crit_filt = np.logical_and(crit_filt, (df_results['Timestamp']<=end_date).values)
    return df_results.loc[(crit_filt)]

# Cell
def get_top(df_results, filt, by, n=5, start_date=None, end_date=None):
    '''
    filter and sort df_results, return top n indices
    optional Timestamp filters
    '''
    idxs=filt_df(df_results, filt).sort_values(by=by, ascending=False).head(n).index.values
    return list(idxs)

# Cell
def get_best_params(eval_conf, filt, by=None, augment=False):
    group_cols = fitting_params if not augment else fitting_params+augment_params
    if by is None: by=eval_conf.m_cols[0]
    grouped = filt_df(eval_conf.df_results, filt).groupby(group_cols, as_index=False)
#     return grouped[[by]].mean()
    return grouped[[by]].mean().sort_values(by=by)


# Cell
def rough_param_comparison(eval_conf, filt, by=None, params=fitting_params):
    if by is None: by=eval_conf.m_cols[0]
    ascending = False if not 'loss' in by else True
    df = filt_df(eval_conf.df_results, filt)
    for param in params:
        print(param)
        print(df.groupby(param)[listify(by)].mean().sort_values(by=by, ascending=ascending))

# Cell
def get_best_params(eval_conf, filt, by=None, augment=False):
    group_cols = fitting_params if not augment else fitting_params+augment_params
    if by is None: by=eval_conf.m_cols[0]
    grouped = filt_df(eval_conf.df_results, filt).groupby(group_cols, as_index=False)
#     return grouped[[by]].mean()
    return grouped[[by]].mean().sort_values(by=by)
