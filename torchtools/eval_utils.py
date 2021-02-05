from torchtools.core import *
from torchtools.data import *
from torchtools.models import *
from torchtools.datasets import *
from torchtools.augmentations import *
from torchtools.dataloader import *
from torchtools.experiments import *
from torchtools.configs import *
from torchtools.evaluation import *

import torch
import pandas as pd
import numpy as np
from fastai.basics import *
import pickle



def _get_mock_learner(ts_experiment, arch):
    #return Learner(ts_runner.db, model=ts_runner.train_params['arch'](ts_runner.db.features, ts_runner.db.c))
    model = get_mod(ts_experiment.dls, arch=arch, dropout=0.1)
    learn = Learner(ts_experiment.dls, model, get_loss_fn('leaky_loss', alpha=0.5))
    return learn

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
                preds_loc='experiments/preds', model_loc='experiments/models', preprocess=True):
        self.is_colab, self.is_class, self.df_base_path = is_colab, is_class, df_base_path
        self.results_loc, self.preds_loc, self.model_loc = results_loc, preds_loc, model_loc
        
    @property
    def base_dir(self): return _get_base_dir(self.is_colab)
    @property
    def preds_dir(self): return self.base_dir/self.preds_loc
    @property
    def results_dir(self): return self.base_dir/self.results_loc
    @property
    def model_dir(self): return self.base_dir/self.model_loc
    @property
    def df_results_path(self): return self.results_dir/_results_fn(self.is_colab, self.is_class)
    @property
    def df_results(self):
        print(self.df_results_path)
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
    
    def reload_df_results(self):
        self._df_results = pd.read_csv(self.df_results_path)
        pre_process_results(self._df_results)


def _reload_model(ts_experiment, eval_conf, idx):
    '''
    load model into ts_experiment
    '''
    fn = eval_conf.df_results.iloc[idx]['model_fn']
    arch = eval_conf.df_results.iloc[idx]['arch']
    ts_experiment.learn = _get_mock_learner(ts_experiment, arch)
    ts_experiment.learn.load(eval_conf.model_dir/Path(fn).stem)
    return

    if not test:
        fn = eval_conf.preds_dir/eval_conf.df_results.iloc[idx]['val_preds'] 
    else:
        fn=eval_conf.preds_dir/eval_conf.df_results.iloc[idx]['test_preds']
    return torch.load(fn)

def load_preds(ts_experiment, eval_conf, idxs, dl_idx=2):
    preds = []
    for idx in idxs:
        _reload_model(ts_experiment, eval_conf, idx)
        preds.append(ts_experiment.learn.get_preds(dl_idx)[0])
    return preds

def complement_idxs(idxs):  
    def _c(idx):
        return idx+1 if idx%2==0 else idx-1
    return tensor([_c(idx) for idx in idxs.numpy()])

def combine_idxs(idxs1, idxs2):
    return list(set(idxs1.numpy()).union(set(idxs2.numpy())))

def combine_idxs_2(idxs1, idxs2):
    return list(set(idxs1.numpy()).intersection(set(idxs2.numpy())))

def _get_bet_idxs(preds, threshold=None, quantile=0.9, complement=False):
    if threshold is None: 
        threshold = np.quantile(preds, quantile) if not complement else -np.quantile(-preds, quantile)
    print(threshold)
    return torch.where(preds>threshold)[0] if not complement else complement_idxs(torch.where(preds<threshold)[0])
