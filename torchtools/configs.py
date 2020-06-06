# AUTOGENERATED! DO NOT EDIT! File to edit: 202_configs.ipynb (unless otherwise specified).

__all__ = ['write_config', 'delete_config', 'read_config', 'config_keys', 'get_simple_config', 'get_discrete_config']

# Cell
import json
from pathlib import Path

# Cell
def write_config(config, config_fn='config.json', overwrite=False):
    '''write a column configuration to json'''

    if not Path(config_fn).is_file():
        print(f'{config_fn} does not exist, creating new config_dict')
        config_dict={}
    else:
        with open(config_fn, 'r') as f:
            config_dict=json.load(f)


    if config['id'] in config_dict.keys() and not overwrite:
        print(f'config with id {config["id"]} already stored, use overwrite=True for overwriting')
        return

    config_dict[config['id']] = config

    with open(config_fn, 'w+') as f:
        json.dump(config_dict, f)

# Cell
def delete_config(config_id, config_fn='config.json'):
    '''delete configuration with id config_id from the configuration dictionary stored in config_fn,
    write it back to disk'''

    if not Path(config_fn).is_file():
        print(f'{config_fn} does not exist')
        return
    else:
        with open(config_fn, 'r') as f:
            config_dict=json.load(f)


    if config_id in config_dict.keys():
        del config_dict[config_id]


#     and not overwrite:
#         print(f'config with id {config["id"]} already stored, use overwrite=True for overwriting')
#         return

#     config_dict[config['id']] = config

    with open(config_fn, 'w+') as f:
        json.dump(config_dict, f)

# Cell
def read_config(conf_id, config_fn='config_json'):
    with open(config_fn, 'r') as f:
        config_dict=json.load(f)
#         print(config_dict)
        try:
            col_config = config_dict[conf_id]
#             print(col_config)
        except:
            print(f'could not load config with id {conf_id}')
    return col_config


# Cell
def config_keys(config_fn='config.json'):
    '''get config ids of a config file'''

    if not Path(config_fn).is_file():
        print(f'{config_fn} does not exist, creating new config_dict')
        config_dict={}
    else:
        with open(config_fn, 'r') as f:
            config_dict=json.load(f)

    return list(config_dict.keys())


# Cell
#copied from datasets
def get_simple_config(discrete=False):
    '''get a simple column configuration for development'''
    if not discrete:
        x_cols = [[f'x{i}_{j}' for j in range(10)] for i in range(6)]
    else:
        x_cols_cont = [[f'x{i}_{j}' for j in range(10)] for i in [0,1,3,4]]
        x_cols_discrete = [[f'x{i}_{j}' for j in range(10)] for i in [2,5]]
        x_cols = x_cols_cont, x_cols_discrete
    dep = 'y0'
    n_train = 8000

    return x_cols, dep, n_train

# Cell
#copied from datablock
def get_discrete_config():
    '''get a simple column configuration for development'''
    x_cols_cont = [[f'x{i}_{j}' for j in range(10)] for i in [0,1,3,4]]
    x_cols_discrete = [[f'x{i}_{j}' for j in range(10)] for i in [2,5]]
    dep = 'y0'
    n_train = 8000

    return x_cols_cont, x_cols_discrete, dep, n_train