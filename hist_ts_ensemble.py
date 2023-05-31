from torchtools.core import *
from torchtools.data import *
from torchtools.models import *
from torchtools.datasets import *
from torchtools.augmentations import *
from torchtools.dataloader import *
from torchtools.experiments import *
from torchtools.configs import *
# from torchtools.evaluation import *
from torchtools.eval_utils import *

import torch
import pandas as pd
import numpy as np
from fastai.basics import *
import pickle
from datetime import timedelta

# df_path = Path('~/coding/python/scrape/bets_processed.csv').expanduser()
df_path = Path('~/coding/python/scrape/bets_historic_20210805.csv.parquet').expanduser()
df_path = Path('~/coding/python/scrape/bets_historic_20210822.csv.parquet').expanduser()
df_path = Path('~/coding/python/scrape/data/bets_processed.csv.parquet').expanduser()
# df_path = Path('~/coding/python/scrape/bets_full_20210716.csv.parquet').expanduser()
COL_CONFIG = 'config2.json'
config_id = 'bets_processed_6c_1yhc'
# config_id = 'bets_tennis_6c_2yahopp'
# config_id = 'bets_enc_16c_2y_ah_opp'

# config_id = 'bets_fb_10c_2y_ah_opp'
# config_id = 'bets_full_20c_2y_ah_opp'
col_config = read_config(config_id, COL_CONFIG)

# n_rows = ts_experiment.df_base.shape[0]
n_rows = pd.read_parquet(df_path).shape[0]
trn_end, val_end, test_end =100000, 100010, n_rows
trn_end, val_end, test_end =500, 510, n_rows
data_params = build_data_params(df_path, col_config=col_config, nrows=None, bs=256, ss_dis=True,
                               trn_end=trn_end, val_end=val_end, test_end=test_end)

fn = 'db_stats.pkl'
with open(fn, 'rb') as f:
    stats = pickle.load(f)
data_params['stats'] = stats
# data_params['stats'] = None
data_params['inference'] = True

ts_experiment = TSExperiments(save_model=False) #can set save model flag here
ts_experiment.setup_data(data_params)
n_rows = ts_experiment.df_base.shape[0]
# test_end = n_rows

is_colab=True,
is_class=False
fn = 'results_colab.csv'
# fn = 'results_colab_202108.csv'
# fn = 'results_colab_bb_opp.csv'
# fn = 'results_colab_opp_202108.csv'
eval_conf = EvalConfig(is_colab, is_class, df_path, preprocess=False, usecols=False, 
        fn=fn)
df_base = ts_experiment.df_base

# idxs = [1685, 1684, 1683, 1682, 1681, 1680]
idxs = [1764, 1758, 1684, 1680] #production
# idxs = [1764, 1758, 1684, 1680, -2] #production
# idxs = [1764, 1758, 1684, 1680, -6, -5, -4, -3, -2]
# idxs = [938]
# idxs = [-3, -2, -1]
preds = load_preds(ts_experiment, eval_conf, idxs, 2, use_best=False)
avg_preds = torch.cat(preds, 1).mean(1)
print('0.8 q', torch.quantile(avg_preds, 0.8), torch.quantile(avg_preds, 0.2))
print('0.85 q', torch.quantile(avg_preds, 0.85), torch.quantile(avg_preds, 0.15))
print('0.9 q', torch.quantile(avg_preds, 0.9), torch.quantile(avg_preds, 0.1))
print('0.95 q', torch.quantile(avg_preds, 0.95), torch.quantile(avg_preds, 0.05))
df_test = df_base.iloc[val_end:test_end].copy()
print(avg_preds)
print(type(avg_preds))
print(avg_preds.shape)
print(df_test.shape)
df_test.loc[:, 'preds'] = avg_preds
df_test.loc[:, 'date'] = pd.to_datetime(df_test.date)
df_test.reset_index(inplace=True, drop=True)
get_opp_preds(df_test)


# df_test.query(f'status=="open" and date>=datetime.utcnow()')[['date', 'horse', 'opponent', 'horse_1x2','preds']].to_csv('/home/johannes/coding/commonresources/matchupinfo/transformer_ensemble.csv')
# df_test.to_csv('/home/johannes/coding/commonresources/matchupinfo/transformer_ensemble_expansion.csv',
#         index=False)
# df_test.to_parquet('/home/johannes/coding/commonresources/matchupinfo/ts_ensemble_full_20210805_2058.parquet')
df_test.to_parquet('/home/johannes/coding/commonresources/matchupinfo/ts_ensemble_soccer_full_20220913.parquet')
# df_test.query('status=="open"')[['date', 'horse', 'opponent', 'horse_1x2','preds']].to_csv('/home/johannes/coding/commonresources/matchupinfo/transformer_ensemble.csv')


ifnone
