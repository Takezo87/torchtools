import json
from pathlib import Path

import pandas as pd

from torchtools.configs import *

#########################
## no feature should be a substring of another feature, e.g. diff and hcp_diff
#########################



PL_1x2, PL_AH, DIFF = 'pl_1x2', 'pl_ah', 'diff'
PL_AH_2, PL_AH_3 = 'pl_ah_2', 'pl_ah_3'
PL_OVER, PL_UNDER, TOTAL_SCORE, HORSE_SCORE, OPPONENT_SCORE = 'pl_over', 'pl_under', 'total_score', 'horse_score', 'opponent_score'
PL_1x2_OPP, PL_AH_OPP, DIFF_OPP = 'pl_1x2_opp', 'pl_ah_opp', 'diff_opp'
PL_AH_2_OPP, PL_AH_3_OPP = 'pl_ah_2_opp', 'pl_ah_3_opp'
PL_HOME_AWAY, PL_HOME_AWAY_OPP = 'pl_home_away', 'pl_home_away_opp'
HCP_DIFF = 'hcp_dff' 
OC_1x2, OC_AH = 'oc_1x2', 'oc_ah'
OC_1x2_OPP, OC_AH_OPP = 'oc_1x2_opp', 'oc_ah_opp'
ODDS_1x2, ODDS_AH, ODDS_HOME_AWAY = 'horse_1x2', 'horse_ah', 'horse_home_away'
ODDS_1x2_OPP, ODDS_AH_OPP, ODDS_HOME_AWAY_OPP = 'horse_1x2_opp', 'horse_ah_opp', 'horse_home_away_opp'
ODDS_1x2_LOG, ODDS_AH_LOG = 'horse_1x2_log', 'horse_ah_log'
FIELD_ENC, HORSE_ENC, OPPONENT_ENC = 'field_enc', 'horse_enc', 'opponent_enc'


COLS_C, COLS_D, COLS_Y, ID = 'cols_c', 'cols_d', 'cols_y', 'id'

COLS_Y2, COLS_Y3 = 'cols_y2', 'cols_y3'

###configuration settings
config_json_fn = 'config_lit.json'


keys=config_keys(config_fn=config_json_fn)

#base df 
# df_base_fn = 'bets_historic_20210822.csv.parquet'
df_base_fn = 'bets_historic_20220103.csv.parquet'
df_base_fn = 'bets_historic_20220117.csv.parquet'
#basketball
# df_base_fn = 'bets_bb_20210821.csv.parquet'
df_base_path = Path('/home/johannes/coding/python/scrape/')/df_base_fn
print(df_base_path)
df_base = pd.read_parquet(df_base_path)

config_name = 'bets_enc_16c_2y_ah_opp'
config_name = 'bb_ts_basic_bets_10c_2y_ah_opp_dummy'
config_name = 'bets_enc_20c_2y_ah_opp'
config_name = 'bets_enc_12c_pl_nooddsoutcomes_2y_ah_opp'
config_name = 'bets_enc_6c_pldiff_2y_ah_opp'
config_name = 'bets_enc_14c_nocats_2y_ah_opp'
config_name = 'bets_enc_20c_2y_ah_opp'
config_name = 'bets_enc_14c_nocats_2y_ah_3_opp'
config_name = 'bets_mult_14c_nocats_2y_ah_2y_ah2_2y_1x2_opp'
config_name = 'bets_enc_14c_nocats_2y_ah_2_opp'
config_name = 'bets_enc_14c_nocats_1y_hcpdiff'
config_name = 'bets_enc_6c_pldiff_2y_ah_2y_1x2_opp'
config_name = 'bets_enc_6c_pldiff_1y_hcpdiff_2y_ah_opp'
config_name = 'bets_enc_8c_diffhcpdiffpl_2y_ah_opp'

#horse on 
# ON_HORSE = [OC_1x2, OC_AH, ODDS_1x2, ODDS_AH, DIFF, FIELD_ENC, HORSE_ENC, OPPONENT_ENC, PL_1x2, PL_AH]
ON_HORSE = [DIFF, PL_1x2, PL_AH]
ON_HORSE = [DIFF, HCP_DIFF, PL_AH, PL_1x2]
# ON_HORSE = [DIFF, FIELD_ENC, HORSE_ENC, OPPONENT_ENC, PL_1x2, PL_AH]
# ON_HORSE = [PL_1x2, PL_AH, DIFF]
#basketball
# ON_HORSE = [PL_HOME_AWAY, PL_AH, ODDS_HOME_AWAY, ODDS_AH, DIFF]
#opponent on 
ON_OPPONENT = []
#horse vs 
VS_HORSE = []
#opponent vs 
# VS_OPPONENT = [OC_1x2, OC_AH, ODDS_1x2, ODDS_AH, DIFF, FIELD_ENC, HORSE_ENC, OPPONENT_ENC,  PL_1x2, PL_AH]
VS_OPPONENT = [DIFF, PL_1x2, PL_AH]
VS_OPPONENT = [DIFF, HCP_DIFF, PL_AH, PL_1x2]
# VS_OPPONENT = [DIFF, FIELD_ENC, HORSE_ENC, OPPONENT_ENC,  PL_1x2, PL_AH]
# VS_OPPONENT = [PL_1x2, PL_AH, DIFF]
#basketball
# VS_OPPONENT = [PL_HOME_AWAY, PL_AH, ODDS_HOME_AWAY, ODDS_AH, DIFF]

##targets
TARGETS = [PL_AH, PL_AH_OPP]
TARGETS = [PL_AH, PL_AH_OPP, PL_AH_2, PL_AH_2_OPP]
TARGETS_MAIN = [PL_AH, PL_AH_OPP]
TARGETS_2 = [PL_AH_2, PL_AH_2_OPP]
TARGETS_3 = [PL_1x2, PL_1x2_OPP]
TARGETS = [PL_AH_2, PL_AH_2_OPP]
TARGETS = [HCP_DIFF]

col_config = {}

h_cols = [[c for c in df_base.columns if c.startswith('f_on_horse') and f in c and '_back' in c] for f in ON_HORSE]
o_cols = [[c for c in df_base.columns if c.startswith('f_on_opponent') and f in c and '_back' in c] for f in ON_OPPONENT]
vh_cols = [[c for c in df_base.columns if c.startswith('f_vs_horse') and f in c and '_back' in c] for f in VS_HORSE]
vo_cols = [[c for c in df_base.columns if c.startswith('f_vs_opponent') and f in c and '_back' in c] for f in VS_OPPONENT]

ts_cols = [*h_cols, *o_cols, *vh_cols, *vo_cols]
col_config[COLS_C] = ts_cols
col_config[COLS_Y] = TARGETS_MAIN
# col_config[COLS_Y2] = TARGETS_MAIN
# col_config[COLS_Y2] = TARGETS_2
# col_config[COLS_Y3] = TARGETS_3
col_config[COLS_D] = None
col_config[ID] = config_name


print([len(c) for c in ts_cols])
for c in ts_cols:
    print(c)
    assert len(c)==10

#for lit ts_basic config, does not work with torchtools
# cols_odds = list(map(ts_cols.__getitem__, [2,3,12,13]))
# cols_odds = list(map(ts_cols.__getitem__, [2,3,9,10])) #14c
cols_odds = list(map(ts_cols.__getitem__, [])) #14c
# cols_outcomes = list(map(ts_cols.__getitem__, [0,1,10,11])) #or profits
# cols_outcomes = list(map(ts_cols.__getitem__, [0,1,7,8])) #or profits
cols_outcomes = list(map(ts_cols.__getitem__, [])) #or profits
cols_diffs = list(map(ts_cols.__getitem__, [0,1,4,5]))
cols_cats = list(map(ts_cols.__getitem__, []))
cols_pl = list(map(ts_cols.__getitem__, [2,3,6,7]))
col_config['cols_odds'] = cols_odds
col_config['cols_outcomes'] = cols_outcomes
col_config['cols_diffs'] = cols_diffs
col_config['cols_cats'] = cols_cats
col_config['cols_pl'] = cols_pl
write_config(col_config, config_fn=config_json_fn, overwrite=False)

print(config_keys(config_json_fn))
