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

# df_path = Path('~/coding/python/scrape/bets_processed_tennis.csv').expanduser()
# df_path = Path('~/coding/python/scrape/bets_processed_tennis.csv.parquet').expanduser()
DATA_DIR = '/home/johannes/coding/python/scrape/data'
df_path = Path(DATA_DIR, 'bets_processed_tennis.csv.parquet')
# df_path = Path('~/coding/python/scrape/bets_historic_tennis.csv').expanduser()
COL_CONFIG = 'config2.json'
config_id = 'bets_tennis_6c_2y_ha_opp'
col_config = read_config(config_id, COL_CONFIG)

# n_rows = pd.read_csv(df_path).shape[0]
n_rows = pd.read_parquet(df_path).shape[0]
val_end, test_end = 500, n_rows
data_params = build_data_params(df_path, col_config=col_config, nrows=None, bs=256, ss_dis=True,
                               trn_end=100, val_end=val_end, test_end=test_end)

fn = 'tennis_6c_80_120_stats.pkl'
with open(fn, 'rb') as f:
    stats = pickle.load(f)
data_params['stats'] = stats
data_params['inference'] = True

ts_experiment = TSExperiments(save_model=False) #can set save model flag here
ts_experiment.setup_data(data_params)

is_colab=True,
is_class=False
eval_conf = EvalConfig(is_colab, is_class, df_path, fn='results_tennis_opp.csv')
# eval_conf = EvalConfig(is_colab, is_class, df_path, fn='results_colab_opp.csv')
df_base = eval_conf.df_base

# idxs = [1685, 1684, 1683, 1682, 1681, 1680]
idxs = [150, 155, 162, 164]
# idxs = [5,6]
preds = load_preds(ts_experiment, eval_conf, idxs, 2)
avg_preds = torch.cat(preds, 1).mean(1)
print(torch.quantile(avg_preds, 0.9), torch.quantile(avg_preds, 0.1))
print(torch.quantile(avg_preds, 0.95), torch.quantile(avg_preds, 0.05))
df_test = df_base.iloc[val_end:test_end].copy()
df_test.loc[:, 'preds'] = avg_preds
df_test.loc[:, 'date'] = pd.to_datetime(df_test.date)

df_test.query('status=="open" and date>=datetime.utcnow()')[['date', 'horse', 'opponent', 'horse_home_away','preds']].to_csv('/home/johannes/coding/commonresources/matchupinfo/ts_tennis.csv')
# df_test.to_csv('/home/johannes/coding/commonresources/matchupinfo/ts_tennis_full.csv')
# df_test.to_csv('/home/johannes/coding/commonresources/matchupinfo/fb_to_tennis_full.csv')
# df_test.query('status=="open"')[['date', 'horse', 'opponent', 'horse_1x2','preds']].to_csv('/home/johannes/coding/commonresources/matchupinfo/transformer_ensemble.csv')


