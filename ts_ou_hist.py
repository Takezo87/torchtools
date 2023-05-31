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

DATA_DIR = '/home/johannes/coding/python/scrape/data'
df_path = Path(DATA_DIR, 'bets_processed_ou.csv.parquet')
# df_path = Path('~/coding/python/scrape/bets_historic_ou.csv').expanduser()
COL_CONFIG = 'config2.json'
config_id = 'bets_processed_12c_2yoverunder'
col_config = read_config(config_id, COL_CONFIG)

out_fn = 'over_bets.csv'

# n_rows = pd.read_csv(df_path).shape[0]
n_rows = pd.read_parquet(df_path).shape[0]
val_end, test_end = 500, n_rows
data_params = build_data_params(df_path, col_config=col_config, nrows=None, bs=256, ss_dis=True,
                               trn_end=100, val_end=val_end, test_end=test_end)

fn = 'ou_18_110_12c_stats.pkl'
with open(fn, 'rb') as f:
    stats = pickle.load(f)
data_params['stats'] = stats
data_params['inference'] = True

ts_experiment = TSExperiments(save_model=False) #can set save model flag here
ts_experiment.setup_data(data_params)

arch = 'transformer_dl'
# fn_model = 'model_pt_True_15_0.0001_best_val.pth'

# preds = load_preds_from_path(ts_experiment, fn_model, arch)
# print(preds.shape)

# eval_conf = EvalConfig(is_colab, is_class, df_path)
# df_base = eval_conf.df_base
is_colab=True,
is_class=False
eval_conf = EvalConfig(is_colab, is_class, df_path, ou=True)
df_base = eval_conf.df_base

# idxs = [1685, 1684, 1683, 1682, 1681, 1680]
idxs = [71, 56, 36, 29]
# idxs = [154, 136, 155, 157]
# idxs = [189, 202, 194, 199, 207]
# idxs = [189, 202, 250, 245, 254]
# idxs = list(range(290, 301))
preds = load_preds(ts_experiment, eval_conf, idxs, 2)
avg_preds = torch.cat(preds, 1).mean(1)
print(torch.quantile(avg_preds, 0.95), torch.quantile(avg_preds, 0.05))
print(torch.quantile(avg_preds, 0.92), torch.quantile(avg_preds, 0.08))
df_test = df_base.iloc[val_end:test_end].copy()
df_test.loc[:, 'preds'] = avg_preds
df_test.loc[:, 'date'] = pd.to_datetime(df_test.date)

# df_test.query('status=="open" and date>=datetime.utcnow()')[[
# # df_test.query('status=="final_result" and date>=datetime(2009,1,1)')[[
#     'date', 'horse', 'opponent', 'horse_1x2','preds']].to_csv(
#             f'/home/johannes/coding/commonresources/matchupinfo/{out_fn}')

df_test.to_csv(
            f'/home/johannes/coding/commonresources/matchupinfo/ts_ou_full_20210610_10.csv')

