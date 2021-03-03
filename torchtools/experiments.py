# AUTOGENERATED! DO NOT EDIT! File to edit: 200_experiments.ipynb (unless otherwise specified).

__all__ = ['df_fn', 'df_dir', 'df_path', 'trn_end', 'val_end', 'test_end', 'splits', 'df_config', 'col_config',
           'df_source', 'dataset_name', 'data_params', 'emb_sz_rule', 'get_emb_sz', 'get_mod', 'get_dls',
           'run_training', 'arch', 'n_epochs', 'max_lr', 'wd', 'loss_fn_name', 'alpha', 'metrics', 'N', 'magnitude',
           'seed', 'pct_start', 'div_factor', 'aug', 'train_params', 'get_recorder_dict', 'TSExperiments',
           'build_data_params']

# Cell
from .core import *
from .data import *
from .models import *
from .datasets import *
from .augmentations import *
#from torchtools.datablock import *
from .dataloader import *
from .configs import *

# Cell
import pandas as pd
import numpy as np
from fastai.basics import *
#from fast_tabnet.core import *
from fastcore.script import *
from fastai.callback.tracker import *

# Cell
from tsai.models.InceptionTimePlus import *
from tsai.models.TSTPlus import *

# Cell
## data config
df_fn = 'bi_sample_anon.csv'
df_dir = Path('./data/custom')
df_path = Path(Path(df_dir)/df_fn)

trn_end = 120000
val_end = 160000
test_end = 200000
splits = (L(range(trn_end)), L(range(trn_end, val_end)))
df_config = f'{int(trn_end/1000)}_{int(val_end/1000)}_{int(test_end/1000)}'

col_config = '6chan_anon_discrete'
cols_c, cols_d, cols_y, n_train = get_discrete_config()

df_source = Path(df_fn).stem


dataset_name = f'{df_source}_{col_config}_{cols_y}_{df_config}'
data_params = defaultdict(lambda:None, {'df_fn':df_fn, 'df_dir':df_dir, 'df_path':df_path, 'trn_end':trn_end,
                                        'val_end':val_end, 'splits':splits, 'col_config_id':col_config,
                                        'cols_c':cols_c, 'cols_d':cols_d, 'cols_y':cols_y, 'ds_id':dataset_name})

# Cell
def emb_sz_rule(n_cat):
    "Rule of thumb to pick embedding size corresponding to `n_cat`"
    return min(600, round(1.6 * n_cat**0.56))

def _one_emb_sz(classes, n, sz_dict=None):
    "Pick an embedding size for `n` depending on `classes` if not given in `sz_dict`."
    sz_dict = ifnone(sz_dict, {})
    n_cat = len(classes[n])
    sz = sz_dict.get(n, int(emb_sz_rule(n_cat)))  # rule of thumb
    return n_cat,sz

def get_emb_sz(to, sz_dict=None):
    "Get default embedding size from `TabularPreprocessor` `proc` or the ones in `sz_dict`"
    return [_one_emb_sz(to.classes, n, sz_dict) for n in to.cat_names]

def get_mod(dls, arch='inception', dropout=None, fc_dropout=None):
    '''
    architectures:
    - inception
    - transformer
    - tst
    - inception_gb, transformer_gb
    '''
    if dls.classification and not dls.mixed:
        return InceptionTime(dls.n_channels, dls.c)

    if arch=='inception_gb': #hack, works only for continuous channels and 1 target
        return InceptionTime(dls.n_channels, 2)

    if arch=='transformer_gb': #hack, works only for continuous channels and 1 target
        return TST(dls.n_channels, 2, 10)

    if arch=='transformer_dl': #hack, works only for continuous channels and exactly 2 targets with double_loss
        #return TST(dls.n_channels, 1, 10):
        return TSTPlus(dls.n_channels, 1, seq_len=10, res_dropout=dropout, fc_dropout=fc_dropout, y_range=(-1,1))

    if dls.n_channels==0:
        assert dls.cols_cat is not None or dls.cols_cont is not None, 'no tabular columns'
        emb_szs= [_one_emb_sz(dls.voc, c) for c in listify(dls.cols_cat)]
        return TabNetTT(emb_szs=emb_szs, n_cont=len(dls.cols_cont), out_sz=dls.n_targets)

    if dls.mixed:
        emb_szs= [_one_emb_sz(dls.voc, c) for c in listify(dls.cols_cat)]

        if dls.classification:
             return InceptionTime_Mixed(dls.n_channels_c, dls.n_channels_d, dls.c,
                                    len(dls.cols_cont), emb_szs=emb_szs)
        else:
            return InceptionTimeD_Mixed(dls.n_channels_c, dls.n_channels_d, dls.n_targets,
                                    len(dls.cols_cont), emb_szs=emb_szs)
    else:
        if dls.dataset.has_x[1]: ##discrete channels
            if arch=='transformer':
                return TransformerSgmD(dls.n_channels, dls.n_targets, res_dropout=dropout)
            else:
                return InceptionTimeD(dls.n_channels, dls.n_targets)
        else:
            if arch=='tst':
                #return TransformerSgm(dls.n_channels, dls.n_targets, res_dropout=dropout)
                return TSTPlus(dls.n_channels, dls.n_targets, seq_len=10, res_dropout=dropout, y_range=(-1,1))
            if arch=='transformer':
                return TransformerSgm(dls.n_channels, dls.n_targets, res_dropout=dropout)
            else:
                return InceptionTimeSgm(dls.n_channels, dls.n_targets)

# Cell
def get_dls(df, cols_c, cols_y, splits, cols_d=None, bs=64, ds_type=TSDatasets5, shuffle_train=True,
           verbose=False, ss_dis=True, cols_cont=None, cols_cat=None, classification=False, stats=None):
    '''
    create dataloaders
    handling of discrete channels with cols_d and ss_dis
    NOTE: continuous tab cols 3d, cat tab cols 2d, legacy....
    '''

    items = items_from_df(df, cols_c, cols_y, len(splits[0]), cols_d=cols_d, tab_cols_c=cols_cont)
    ars=items_to_arrays(items)
    has_col=[cols_c is not None, cols_d is not None, cols_cont is not None]
    Xc, Xd, X_conts = map_xs(ars[:-1], has_col)

    y=ars[-1].astype(np.float)
    if classification:
        y, y_vocab = cats_from_df(df, listify(cols_y), len(splits[0]), add_na=False)
        y=y.squeeze()
        y=y.astype(np.long)

    if cols_cat is not None:
        X_cats, cat_maps = cats_from_df(df, cols_cat, len(splits[0]))
    else: X_cats, cat_maps = None, None

    _ytype=TensorCategory if classification else TensorFloat
    print(ds_type)
    dsets = ds_type(X_c=Xc, X_d=Xd, y=y, splits=splits, X_tcont=X_conts, X_tcat=X_cats, _ytype=_ytype)
#     dsets = ds_type(X=Xc, X_dis=Xd, y=y, splits=splits, X_tabc=X_conts, X_tabcat=X_cats, _ytype=_ytype)
    print(dsets.n_subsets)

    ##standardization: continuous channels always, discrete channels optional
    batch_tfms=[]
    print(has_col)
    ###!!!!HACK!!!!!
    if stats is not None: batch_tfms+=[TSStandardize(by_var=True, verbose=verbose).from_stats(*stats)]
    else:
        if has_col[0] and ss_dis: batch_tfms+=[TSStandardize(by_var=True, verbose=verbose)]
        if has_col[1] and ss_dis: batch_tfms+=[TSStandardize(by_var=True, verbose=verbose, discrete=True)]
#     augmix = AugmixSS()
#     print(batch_tfms)
#     return dsets
    ds = [dsets.subset(i) for i in range(dsets.n_subsets)]
#     return ds
#     dls = TSDataLoaders.from_dsets(*ds, bs=[bs]+[bs]*len(splits), batch_tfms=batch_tfms, shuffle_train=shuffle_train)
    dls = TSDataLoaders.from_dsets(*ds, bs=bs, batch_tfms=batch_tfms, shuffle_train=shuffle_train)
#     dls = TSDataLoaders.from_dsets(dsets.train, bs=[128,128])
    dls.n_channels = len(listify(cols_c)) + len(listify(cols_d))
    dls.n_channels_c = len(listify(cols_c))
    dls.n_channels_d = len(listify(cols_d))

    dls.n_targets = len(listify(cols_y))
    dls.cols_cat, dls.cols_cont = cols_cat, cols_cont
    if cols_cat is not None:
        dls.voc=cat_maps

    dls.mixed = dls.cols_cat is not None or dls.cols_cont is not None

    if classification:
        dls.y_vocab=y_vocab
        dls.c = len(dls.y_vocab[list(dls.y_vocab.keys())[0]])
    dls.classification = True if classification else False
    ##ToDO: for mixed input, store category info in dl

    return dls

# Cell
def _remove_augs(dls):
    '''
    remove augmentation transforms from dls.after_batch
    '''
    fs = [f for f in dls.after_batch.fs if not issubclass(type(f), AugTransform)]
    print(fs)
    dls.after_batch.fs.clear()
    for f in fs: dls.after_batch.add(f)
    print(dls.after_batch, dls.train.after_batch)

# Cell
def run_training(dls, arch=None, seed=1234, n_epochs=None, max_lr=None, wd=None,
                 loss_fn_name=None, alpha=None, metrics=unweighted_profit,
                 N=2, magnitude=0.1, pct_start=0.3, div_factor=25.0, aug='randaugment', **kwargs):
    # model = ResNetSig(db.features, db.c).to(device)
    '''
    run a training cycle
    parameterization important for keeping track

    '''
    assert loss_fn_name and n_epochs, 'must pass loss_fn_name, and n_epochs'

    print(f'pct_start: {pct_start} div_factor: {div_factor}')
    set_seed(seed)
#     model = arch(db.features, db.c)
#     model = arch(6,1)

    model = get_mod(self.dls, arch=arch)

    _remove_augs(dls)
    augs = RandAugment(N=N, magnitude=magnitude, verbose=True) if aug=='randaugment' else Augmix(
        N=N, magnitude=magnitude, verbose=True) if aug=='augmix' else None
#     augs  = Augmix(verbose=True)
    if augs: dls.after_batch.add(augs)
    loss_fn = get_loss_fn(loss_fn_name, alpha=alpha) if not dls.classification else get_loss_fn
    print(loss_fn)

    learn = Learner(dls, model, loss_func=loss_fn, metrics=metrics)

    learn.fit_one_cycle(n_epochs, max_lr, wd=wd, pct_start=pct_start, div_factor=div_factor)
#     learn.recorder.plot_losses()
#     learn.recorder.plot_metrics()
    return learn

# Cell
#train params
arch = InceptionTimeD
n_epochs = 5
max_lr = 1e-5
wd = 0.03
loss_fn_name = 'leaky_loss'
alpha = 0.5
metrics = [unweighted_profit]#, partial(unweighted_profit, threshold=0.2),
           #partial(unweighted_profit, threshold=0.5)] #[weighted_profit, unweighted_profit_0, unweighted_profit_05]
N = 3
magnitude = 0.4
# bs = [64*4, 64*4*2]  treated as a data_param
# y_range = (-1, 1) # not sure yet about this one
seed = 1234
# ds_name = dataset_name #data_param inferred
# ds_path = str(ds_full_path) #data_param
pct_start=0.3                   #fastai default: 0.3
div_factor = 25.0               #fastai default 25.0
aug='augmix'

#default dict?
train_params = {'arch':arch, 'n_epochs':n_epochs, 'max_lr':max_lr, 'wd':wd, 'loss_fn_name':loss_fn_name, 'alpha':alpha,
               'metrics':metrics, 'N':N, 'magnitude':magnitude,
                #'bs':bs,
                'seed':seed,
                #'ds_name':ds_name,
               'pct_start':pct_start, 'div_factor':div_factor, 'aug':aug}

# Cell
def _losses_from_recorder(r, metrics=False):
    idx = slice(0,2) if not metrics else slice(2,None)
    return r.values[-1][idx]
def _minmax_values_from_recorder(r, metrics=False):
    idx = [0,1] if not metrics else list(range(2, len(r.values[0])))
    f = np.min if not metrics else np.max
    return L([f(L(r.values).itemgot(i)) for i in idx])

# Cell
def get_recorder_dict(recorder):
    '''
    return a dictionary containing train and validation loss and metrics values
    '''
    metrics = recorder.metrics
#     loss_values = [recorder.losses[-1].item(), recorder.val_losses[-1].item()]
    loss_values = _losses_from_recorder(recorder)
#     loss_min_values = [np.min(recorder.losses), np.min(recorder.val_losses)]
    loss_min_values = _minmax_values_from_recorder(recorder)
    metrics_values = _losses_from_recorder(recorder, metrics=True)
#     metrics_max_values = [np.max([m[i] for m in recorder.metrics]) for i in range(len(recorder.metrics[0]))]
    metrics_max_values = _minmax_values_from_recorder(recorder, metrics=True)
    recorder_keys = ['trn_loss', 'val_loss', 'trn_loss_min', 'val_loss_min',
                     *[f'{m.name}_{i}_value' for i,m in enumerate(metrics)], *[f'{m.name}_{i}_max' for i,m in enumerate(metrics)]]
    return dict(zip(recorder_keys, loss_values+loss_min_values+metrics_values+metrics_max_values))

# Cell
def _to_flat_dict(train_params):
    flat_dict={}
    for key,value in train_params.items():
        if key=='metrics':
            for i,_ in enumerate(listify(value)):
                flat_dict[f'metric_{i}'] = value[i].__name__
        #arch parameter should be string, but used to be <class model>
        elif key=='arch' and not isinstance(value, str): flat_dict[key] = value.__name__
        else: flat_dict[key]=value
    return flat_dict

# Cell
def _write_results(df, fn):
    if not os.path.isfile(fn):
        df.to_csv(fn, index=False)
    else:
        print('not new')
        df_old = pd.read_csv(fn)
        df_new = pd.concat([df_old, df], ignore_index=True, sort=False)
#         df.to_csv(fn, index=False, mode='a', header=False)
        df_new.to_csv(fn, index=False)

# Cell
def _dict_product(params):
            values = list(itertools.product(*params.values()))
            return [dict(zip(params.keys(), values[i])) for i in range(len(values))]

# Cell
def _get_preds_fn(prefix='val'):
    return f'{prefix}_preds_{abs(hash(datetime.utcnow()))}.pt'

def _get_model_fn(prefix='model'):
    return f'{prefix}_{abs(hash(datetime.utcnow()))}'

# Cell
def _id_from_splits(splits):
    return '_'.join([(str((l[-1]+1)//1000)) for l in splits])

def _get_ds_id(data_params, splits):
    return f"{data_params['df_path'].stem}_{data_params['col_config_id']}_{_id_from_splits(splits)}"


# Cell
class TSExperiments:
    '''
    Wrapper class for Timeseries modelling experiment
    needed: data_params, train_params for setup
    provides:
        - `run_experiment(df_results)`: run one modelling run using `train_params`
        - `grid_search(hypers, df_results)`: update `train_params` with each possible configuration of `hypers`
        and run the respective experiment
    experimental results and all necessary parameters for reproducibility is stored in `df_results`

    '''
    def __init__(self, save_model=False, preds_path=None, model_path=None, results_path=None):
        ##reproducibility
        torch.backends.cudnn.deterministic = True

        #self.train_params = train_params #training params can change, e.g. when running grid search
        self.save_model = save_model## models are big
        self.preds_path = ifnone(preds_path, './experiments/preds')
        self.model_path = ifnone(model_path, './experiments/models')
        self.results_path = ifnone(results_path, './experiments/results')

    def setup_data(self, data_params):
        #read in dataframe
        self.data_params=data_params
        self.df_base = pd.read_csv(data_params['df_path'], nrows=data_params['nrows'])

        #get continuous, discrete, and dependent columns
        cols_c, cols_d, cols_y, splits, ss_dis = map(data_params.get, ['cols_c', 'cols_d', 'cols_y', 'splits', 'ss_dis'])
        cols_cat, cols_cont= map(data_params.get, ['cols_cat', 'cols_cont']) ## tabular data

        prune = data_params.get('prune', None)
        if prune is not None:
            assert prune in ['hcodds_col', 'overodds_col']
            prune_col = data_params.get(prune)
            print(prune_col)
            assert prune_col is not None, 'prune value has to be a valid columns'
            self.df_base.drop(self.df_base[self.df_base[prune_col]<=1].index, inplace=True)
            self.df_base.reset_index(inplace=True, drop=True)

        #get splits
#         print(splits, callable(splits))
        self.splits = splits(self.df_base) if callable(splits) else splits
#         print(self.splits)

        ##store some of the data parameters for later use
        self.bs = data_params['bs']
        self.ds_id = _get_ds_id(data_params, self.splits)
        self.classification = data_params.get('classification', False)
        self.prune = prune
        self.stats = data_params.get('stats')
#         self.dls = get_dls(self.df_base, cols_c, cols_y, self.splits, cols_d=cols_d, bs=self.bs,
#                            ss_dis=ss_dis)
        self.dls = get_dls(self.df_base, cols_c, cols_y, self.splits, cols_d=cols_d, bs=self.bs,
                           ss_dis=ss_dis, cols_cont=cols_cont, cols_cat=cols_cat, ds_type=TSDatasets5,
                          classification=self.classification, stats=self.stats)


    def setup_training(self, train_params):
        assert hasattr(self, 'data_params'), 'setup_data first'
        self.train_params = train_params
        self.train_params['bs']=self.bs
        self.train_params['ds_id'] = self.ds_id
        self.train_params['classification'] = self.classification
        self.train_params['prune'] = self.prune

        if self.train_params['classification']:
            assert self.train_params['loss_fn_name'] in ["crossentropy", "rww"]


    def _save_preds(self, test=False):
    #         val_preds_fn = _get_preds_fn()
        preds_fn = _get_preds_fn()
        preds, y_true = self.learn.get_preds(1)
        torch.save(preds, Path(self.preds_path)/preds_fn)
        self.df_dict.update({'val_preds':preds_fn})
        if len(list(self.dls))==3:
            preds_fn = _get_preds_fn('test')
            preds, y_true = self.learn.get_preds(2)
            torch.save(preds, Path(self.preds_path)/preds_fn)
            self.df_dict.update({'test_preds':preds_fn})

    def _save_model(self):
    #         val_preds_fn = _get_preds_fn()
        model_fn = self.model_fn
        self.learn.save(model_fn)
        self.df_dict.update({'model_fn':f'{self.learn.model_dir}/{model_fn}.pth'})



    def run_training(self, arch=None, seed=1234, n_epochs=None, max_lr=None, wd=None,
                     loss_fn_name=None, alpha=None, metrics=unweighted_profit,
                     N=2, magnitude=0.1, pct_start=0.3, div_factor=25.0, aug='randaugment',
                     verbose=False, weight=None, save_best=False, **kwargs):
        # model = ResNetSig(db.features, db.c).to(device)
        '''
        run a training cycle
        parameterization important for keeping track
        '''
        assert loss_fn_name and n_epochs, 'must pass loss_fn_name, and n_epochs'

        print(f'pct_start: {pct_start} div_factor: {div_factor}')

        ## reset dls.rng --> consistent shuffling

#         huffle_fn(self, idxs): return self.rng.sample(idxs, len(idxs))
#         print(self.)
    #     model = arch(db.features, db.c)


        _remove_augs(self.dls)
        if aug=='randaugment':  augs=RandAugment(N=N, magnitude=magnitude, verbose=verbose)
#         elif aug=='augmix': augs=Augmix(N=N, magnitude=magnitude, verbose=verbose)
        elif aug=='augmix':
            _augmixtype=AugmixSS if kwargs.get('augmixss') is not None else Augmix
            augs=_augmixtype(N=N, magnitude=magnitude, verbose=verbose)
            print(f'augmix order {augs.order}')
        else:
            print(f'no augmentation with value {aug}')
            augs=None
        print(augs)
        print(augs is None)
        if augs:
            self.dls.after_batch.add(augs)
            augs.setup(self.dls[0])
            ## Pipeline.add does not reorder the transforms, but we want the augmentation before the standardisation
            self.dls.after_batch.fs = self.dls.after_batch.fs.sorted(key='order')
        cbs = [SaveModelCallback(fname=f'{self.model_fn}_best_val'),
               #SaveModelCallback(fname=f'{self.model_fn}_best_combo_profit', monitor='combo_profit'),
              ] if save_best else None

#         loss_fn = get_loss_fn(loss_fn_name, alpha=alpha)
        loss_fn = get_loss_fn(loss_fn_name, alpha=alpha) if not self.dls.classification else get_loss_fn_class(
            loss_fn_name, weight=weight)
        print(loss_fn)


        set_seed(seed)
        self.dls.train.rng = random.Random(random.randint(0,2**32-1))

#         model = arch(self.dls.n_channels, self.dls.n_targets)
        model = get_mod(self.dls, arch=self.train_params['arch'], dropout=self.train_params.get('dropout'),
                       fc_dropout=self.train_params.get('fc_dropout'))
        learn = Learner(self.dls, model, loss_func=loss_fn, metrics=metrics, model_dir=self.model_path,
                       wd=wd, cbs=cbs)
        print(learn.dls.after_batch)

#         print(f'wd: {wd} {learn.wd}')
        learn.fit_one_cycle(n_epochs, max_lr, wd=wd, pct_start=pct_start, div=div_factor)
#         learn.fit_one_cycle(n_epochs, max_lr, wd=wd)
    #     learn.recorder.plot_losses()
    #     learn.recorder.plot_metrics()
        return learn


    def run_experiment(self, df_fn=None):
        '''
        could wrap the dataset parameters
        '''
        assert df_fn is not None, 'please specify results csv filename'
        self.model_fn = _get_model_fn()

        self.learn = self.run_training(**self.train_params)
#         rec_dict = get_recorder_dict(self.learn.recorder)
        self.df_dict = dict()
        self.df_dict.update(_to_flat_dict(train_params))
        self.df_dict.update(get_recorder_dict(self.learn.recorder))
        self.df_dict['Timestamp'] = str(datetime.now())
        ## store prediction in a separate file as tensors, but add filename
        self._save_preds(test=False)
        if self.save_model: self._save_model()
        if self.save_model or self.train_params.get('save_best'):
            self.df_dict.update({'model_fn':f'{self.learn.model_dir}/{self.model_fn}.pth'})
        _write_results(pd.DataFrame([self.df_dict], index=[0]), Path(self.results_path)/df_fn)
#         return df_dict



    def run_grid_search(self, hypers:dict, df_results_fn=None):
        '''
        run hyper parameter grid search, note that this changes self.train_params
        '''
        hyper_configs = _dict_product(hypers) #list of dictionaries
        if hasattr(self, 'hyper_configs'): self.hyper_configs+=hyper_configs
        else: self.hyper_configs = hyper_configs
        for config in hyper_configs:
            self.train_params.update(config)
            print(self.train_params)
            self.run_experiment(df_fn=df_results_fn)

# Cell
def build_data_params(df_path, trn_end=None, val_end=None, test_end=None, splitter_fn=TSSplitter(),
                      col_config=None, col_fn=None, bs=64, nrows=None, ss_dis=True, classification=False):
#     assert col_config or col_fn, 'need to pass either cont. cols and y cols, or a col_fn'

    assert col_config, 'need to pass columns configuration'

    if trn_end and val_end:
        splits=L(L(range(trn_end)), L(range(trn_end, val_end)))
        if test_end: splits.append(L(range(val_end, test_end)))
    else:
        splits = splitter_fn

    cols_c, cols_d, cols_y, cols_config_id, cols_cont, cols_cat, prune, hcodds_col = map(
        col_config.get, ['cols_c', 'cols_d', 'cols_y', 'id', 'cols_cont', 'cols_cat', 'prune', 'hcodds_col'])

#     dataset_name = f'{df

    data_params = defaultdict(lambda:None, {'df_path':df_path, 'splits':splits, 'col_config_id':cols_config_id,
                                            'cols_c':cols_c, 'cols_d':cols_d, 'cols_y':cols_y, 'cols_cont':cols_cont,
                                             'cols_cat':cols_cat, 'hcodds_col': hcodds_col, 'bs':bs, 'prune':prune,
                                            'nrows':nrows, 'ss_dis':ss_dis,'classification':classification})
#                'ds_full_path':ds_full_path,
                 #'dataset_name':dataset_id,

    return data_params

# Cell
def _get_arch(arch:str, with_discrete=False):
    if arch.lower()=='inception': return InceptionTimeSgm if not with_discrete else InceptionTimeD
    elif arch.lower()=='resnet': return 'ResNet not implemented'
    else: return None