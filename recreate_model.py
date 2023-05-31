from torchtools.experiments import *
from torchtools.evaluation import *
from torchtools.configs import *

import pandas as pd
from pathlib import Path

results_fn = '~/google-drive/experiments/results/results_colab.csv'
COL_CONFIG = 'config2.json'
df_path = Path('~/google-drive/data/bets_anon_new.csv').expanduser()
df_results_rc = 'results_recreate.csv'

idx = 533

df_results = pd.read_csv(results_fn)


train_params.update(dict(df_results.iloc[idx][augment_params]))
train_params.update(dict(df_results.iloc[idx][fitting_params]))
train_params.update(dict(df_results.iloc[idx][['seed', 'arch', 'alpha']]))
train_params.update({'metrics': metric_cols})
train_params['arch'] = 'inception'

print(train_params)



# def splits_from_ds_id(df_results, idx):
#     ds_id = df_results.iloc[idx]['ds_id']
#     parts = 

train_end, val_end, test_end = 160000, 185000, 210000
config_id = 'anon_10sl_4c_2d_1yhc'
col_config = read_config(config_id, COL_CONFIG)

bs = df_results.iloc[idx]['bs']

data_params = build_data_params(df_path, trn_end, val_end, test_end, col_config=col_config, bs=bs)

ts_experiment = TSExperiments()
ts_experiment.setup_data(data_params)

ts_experiment.setup_training(train_params)

ts_experiment.run_experiment(df_results_rc)

