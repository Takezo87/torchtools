from torchtools.core import *
from torchtools.data import *
from torchtools.models import *
from torchtools.datasets import *
from torchtools.augmentations import *
from torchtools.dataloader import *
from torchtools.experiments import *
from torchtools.configs import *
# from torchtools.evaluation import *

import torch
import pandas as pd
import numpy as np
from fastai.basics import *
import pickle
import os


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

def _get_mock_learner(ts_experiment, arch):
    #return Learner(ts_runner.db, model=ts_runner.train_params['arch'](ts_runner.db.features, ts_runner.db.c))
    print(arch)
    model = get_mod(ts_experiment.dls, arch=arch, dropout=0.1, fc_dropout=0.1)
    # print(model)
    if arch=='transformer_dl':
        learn = Learner(ts_experiment.dls, model, get_loss_fn('double_loss_squared', alpha=0.5))
    else:
        learn = Learner(ts_experiment.dls, model, get_loss_fn('leaky_loss', alpha=0.5))
    return learn

def _get_base_dir(is_colab=False):
    return Path('./') if not is_colab else Path('~/google-drive').expanduser()

def _results_fn(is_colab=False, is_class=False, ou=False):
    if is_colab:
        results_fn='results_colab.csv' if not is_class else 'results_colab_class.csv'
        if ou: results_fn='results_colab_ou.csv'
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
                preds_loc='experiments/preds', model_loc='experiments/models', preprocess=True,
                ou=False, fn=None, usecols=False):
        self.is_colab, self.is_class, self.df_base_path, self.ou = is_colab, is_class, df_base_path, ou
        self.results_loc, self.preds_loc, self.model_loc, self.fn = results_loc, preds_loc, model_loc, fn
        self.preprocess = preprocess
        self.usecols = usecols
        
    @property
    def base_dir(self): return _get_base_dir(self.is_colab)
    @property
    def preds_dir(self): return self.base_dir/self.preds_loc
    @property
    def results_dir(self): return self.base_dir/self.results_loc
    @property
    def model_dir(self): return self.base_dir/self.model_loc
    @property
    def df_results_path(self):
        fn = self.fn if self.fn is not None else _results_fn(self.is_colab, self.is_class, self.ou)
        return self.results_dir/fn
    @property
    def df_results(self):
        print(self.df_results_path)
        if not hasattr(self, '_df_results'): 
            self._df_results = pd.read_csv(self.df_results_path)
            if self.preprocess:
                pre_process_results(self._df_results)
        return self._df_results
            
    @property
    def df_base(self):
        print(self.df_base_path)
        cols = ['pl_ah', 'pl_ah_opp', 'pl_1x2', 'pl_1x2_opp']
        if not self.usecols: cols=None
        if not hasattr(self, '_df_base'):
            # self._df_base = pd.read_csv(self.df_base_path, usecols=cols)
            self._df_base = pd.read_parquet(self.df_base_path, columns=cols)
        return self._df_base
    @property
    def m_cols(self):
        return metric_cols_classification if self.is_class else metric_cols
    
    def reload_df_results(self):
        self._df_results = pd.read_csv(self.df_results_path)
        pre_process_results(self._df_results)

    def delete_row(row_idx):
        '''
        delete experiment row and associated models, predictions
        '''
        pass


def delete_model(eval_conf, idx):
    fn = eval_conf.df_results.iloc[idx]['model_fn']
    print(fn)
    os.remove(eval_conf.model_dir/Path(fn).name)

def _reload_model(ts_experiment, eval_conf, idx, use_best=False):
    '''
    load model into ts_experiment
    '''
    fn = eval_conf.df_results.iloc[idx]['model_fn']
    # print('fn')
    arch = eval_conf.df_results.iloc[idx]['arch']
    print(arch)
    ts_experiment.learn = _get_mock_learner(ts_experiment, arch)
    if not use_best:
        ts_experiment.learn.load(eval_conf.model_dir/Path(fn).stem)
    else:
        ts_experiment.learn.load(eval_conf.model_dir/(Path(fn).stem+'_best_val'))
    return

    if not test:
        fn = eval_conf.preds_dir/eval_conf.df_results.iloc[idx]['val_preds'] 
    else:
        fn=eval_conf.preds_dir/eval_conf.df_results.iloc[idx]['test_preds']
    print(fn)
    return torch.load(fn)

def _reload_model_from_path(ts_experiment, fn, arch):
    '''
    load model into ts_experiment, full path
    '''
    ts_experiment.learn = _get_mock_learner(ts_experiment, arch)
    # print(ts_experiment.learn.model)
    ts_experiment.learn.load(Path(fn).stem)

def load_preds(ts_experiment, eval_conf, idxs, dl_idx=2, use_best=False):
    preds = []
    for idx in idxs:
        _reload_model(ts_experiment, eval_conf, idx, use_best=use_best)
        #fix, with fastcore 1.3.20 and fastai 2.3.1, get_preds removes TSStandardize from the
        #dataloader
        tsstandardize = ts_experiment.dls[dl_idx].after_batch[0] #more than one transform?
        # dl = ts_experiment.dls[dl_idx]
        p, y = ts_experiment.learn.get_preds(dl_idx)
        # p, y = ts_experiment.learn.get_preds(dl=dl)
        ts_experiment.dls[dl_idx].after_batch.add(tsstandardize)
        assert len(ts_experiment.dls[dl_idx].after_batch.fs)==1
        # print(ts_experiment.dls[dl_idx].after_batch[0].mean)
        print(torch.quantile(p, 0.95))
        preds.append(p)
    return preds

def load_preds_from_path(ts_experiment, fn, arch, dl_idx=2):
    _reload_model_from_path(ts_experiment, fn, arch)
        # preds.append(ts_experiment.learn.get_preds(dl_idx)[0])
    preds, _ = ts_experiment.learn.get_preds(dl_idx)
    return preds

def complement_idxs(idxs):  
    '''pandas index input'''
    def _c(idx):
        return idx+1 if idx%2==0 else idx-1
    if isinstance(idxs, pd.Index):
        return tensor([_c(idx) for idx in idxs.numpy()])
    else:
        return tensor([_c(idx) for idx in idxs])

def combine_idxs(idxs1, idxs2):
    return list(set(idxs1.numpy()).union(set(idxs2.numpy())))

def combine_idxs_2(idxs1, idxs2):
    return list(set(idxs1.numpy()).intersection(set(idxs2.numpy())))

def _get_bet_idxs(preds, threshold=None, quantile=0.9, complement=False):
    if threshold is None: 
        threshold = np.quantile(preds, quantile) if not complement else -np.quantile(-preds, quantile)
    print(threshold)
    return torch.where(preds>threshold)[0] if not complement else complement_idxs(torch.where(preds<threshold)[0])

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

def get_opp_preds(df, preds_col='preds'):
    '''
    create a column for the opponent predictions
    '''
    idxs_c = complement_idxs(tensor(df.index))
    df['preds_opp'] = df.iloc[idxs_c][preds_col].values
   
def eval_ou_df(df, q=0.95, min_date=datetime(2010,1,1), by_year=False):
    df.date = pd.to_datetime(df.date)
    th, td = np.quantile(df.preds.values,q) , np.quantile(df.preds.values, 1-q)

    print(df.loc[np.logical_and(df.preds>=th, df.date>=min_date)][['pl_over', 'pl_under']].agg(['mean', 'sum', 'count']))
    print(df.loc[np.logical_and(df.preds<=td, df.date>=min_date)][['pl_over', 'pl_under']].agg(['mean', 'sum', 'count']))


def basic_eval(eval_conf, model_idx, test=False, threshold=None, quantile=0.9, complement=False,
        target_cols=['y0', 'y1'], end_train=170000, end_valid=200000, end_test=230000):
    '''
    reload model prediction for one or more(ensemble) model indices and do a basic evaluation
    '''
    if is_listy(model_idx):
        preds = _average_preds(eval_conf, model_idx, test=test)
    else:
        preds = _reload_preds(eval_conf, model_idx, test=test)
   
    if test:
        splits=[L(range(end_train)), L(range(end_valid, end_test))] ##TODO put into eval_conf
    else:
        splits=[L(range(end_train)), L(range(end_train, end_valid))]
        
    bet_idxs = _get_bet_idxs(preds, threshold, quantile, complement=complement)
    print(eval_conf.df_base.iloc[splits[1]].iloc[bet_idxs][target_cols].agg(['mean', 'sum', 'count']))
    if is_listy(model_idx):
        preds_all = [_reload_preds(eval_conf, idx, test=test) for idx in model_idx]
        bet_idxs_all = [_get_bet_idxs(preds, threshold, quantile, complement=complement) for preds in preds_all]
        res_all = [eval_conf.df_base.iloc[splits[1]].iloc[bet_idxs][[target_cols[0]]].mean().values for bet_idxs in bet_idxs_all]
        print(f'single results {res_all}')
        print(f'mean result {np.array(res_all).mean()}')
    return bet_idxs

def complement_idxs(idxs, ou=False):
    if ou: return idxs
    def _c(idx):
        return idx+1 if idx%2==0 else idx-1
    return tensor([_c(idx) for idx in idxs.numpy()])

def combine_idxs(idxs1, idxs2):
    return list(set(idxs1.numpy()).union(set(idxs2.numpy())))

def combine_idxs_2(idxs1, idxs2):
    return list(set(idxs1.numpy()).intersection(set(idxs2.numpy())))

def _get_bet_idxs(preds, threshold=None, quantile=0.9, complement=False, ou=False):
    if threshold is None: 
        threshold = np.quantile(preds, quantile) if not complement else -np.quantile(-preds, quantile)
    print(threshold)
    return torch.where(preds>threshold)[0] if not complement else complement_idxs(torch.where(preds<threshold)[0], ou=ou)



def basic_eval_ou(eval_conf, model_idx, test=False, threshold=None, quantile=0.9, complement=False,
                 end_train=80000, end_val=110000, pl_cols=['pl_over', 'pl_under']):
    '''
    reload model prediction for one or more(ensemble) model indices and do a basic evaluation
    '''
    if is_listy(model_idx):
        preds = _average_preds(eval_conf, model_idx, test=test)
    else:
        preds = _relead_preds(eval_conf, model_idx, test=test)
   
    if test:
        splits=[L(range(end_train)), L(range(end_train,end_val))] ##TODO put into eval_conf
    else:
        splits=[L(range(end_train)), L(range(end_train,end_val))]
        
    bet_idxs = _get_bet_idxs(preds[:, 0], threshold, quantile, complement=complement, ou=True)
    print(eval_conf.df_base.iloc[splits[1]].iloc[bet_idxs][pl_cols].agg(['mean', 'sum', 'count']))
    
    #print(eval_conf.df_base.iloc[splits[1]].iloc[bet_idxs][['pl_ah', 'pl_ah_opp']].agg(['mean', 'sum', 'count']))
    if is_listy(model_idx):
        preds_all = [_reload_preds(eval_conf, idx, test=test) for idx in model_idx]
        bet_idxs_all = [_get_bet_idxs(preds, threshold, quantile, complement=complement, ou=True) for preds in preds_all]
        res_all = [eval_conf.df_base.iloc[splits[1]].iloc[bet_idxs][[pl_cols[0]]].mean().values for bet_idxs in bet_idxs_all]
        #res_all = [eval_conf.df_base.iloc[splits[1]].iloc[bet_idxs][['pl_ah']].mean().values for bet_idxs in bet_idxs_all]
        print(f'single results {res_all}')
        print(f'mean result {np.array(res_all).mean()}')
    return bet_idxs

def basic_tennis_opp(eval_conf, model_idx, test=False, threshold=None, quantile=0.9, complement=False):
    '''
    reload model prediction for one or more(ensemble) model indices and do a basic evaluation
    '''
    if is_listy(model_idx):
        preds = _average_preds(eval_conf, model_idx, test=test)
    else:
        preds = _reload_preds(eval_conf, model_idx, test=test)
   
    if test:
        splits=[L(range(80000)), L(range(80000,120000))] ##TODO put into eval_conf
    else:
        splits=[L(range(80000)), L(range(80000,120000))]
        
    bet_idxs = _get_bet_idxs(preds[:, 0], threshold, quantile, complement=complement, ou=True)
    #print(eval_conf.df_base.iloc[splits[1]].iloc[bet_idxs][['pl_over', 'pl_under']].agg(['mean', 'sum', 'count']))
    print(eval_conf.df_base.iloc[splits[1]].iloc[bet_idxs][['pl_home_away', 'pl_home_away_opp', 'pl_ah', 'pl_ah_opp']].agg(['mean', 'sum', 'count']))
    if is_listy(model_idx):
        preds_all = [_reload_preds(eval_conf, idx, test=test) for idx in model_idx]
        bet_idxs_all = [_get_bet_idxs(preds, threshold, quantile, complement=complement, ou=True) for preds in preds_all]
        #res_all = [eval_conf.df_base.iloc[splits[1]].iloc[bet_idxs][['pl_over']].mean().values for bet_idxs in bet_idxs_all]
        res_all = [eval_conf.df_base.iloc[splits[1]].iloc[bet_idxs][['pl_home_away']].mean().values for bet_idxs in bet_idxs_all]
        print(f'single results {res_all}')
        print(f'mean result {np.array(res_all).mean()}')
    return bet_idxs
