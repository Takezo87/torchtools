# AUTOGENERATED! DO NOT EDIT! File to edit: 30_datasets.ipynb (unless otherwise specified).

__all__ = ['To3dArray', 'ToArray', 'decompress_from_url', 'get_UCR_univariate_list', 'get_UCR_multivariate_list',
           'get_UCR_univariate', 'get_UCR_multivariate', 'get_UCR_data', 'ucr_to_items', 'get_simple_config',
           'items_from_df', 'cats_from_df']

# Cell
from .data import *

# Cell
import numpy as np
#import torch
from fastai.torch_basics import *
from fastai.data.all import *
from fastai.callback.all import *
import torch

# Cell
import os
import tempfile
try: from urllib import urlretrieve
except ImportError: from urllib.request import urlretrieve
import shutil
from pyunpack import Archive
from scipy.io import arff

# Cell
#TSUtilities
def To3dArray(arr):
    arr = ToArray(arr)
    if arr.ndim == 1: arr = arr[None, None]
    elif arr.ndim == 2: arr = arr[:, None]
    elif arr.ndim == 4: arr = arr[0]
    assert arr.ndim == 3, 'Please, review input dimensions'
    return np.array(arr)

def ToArray(arr):
    if isinstance(arr, torch.Tensor):
        arr = np.array(arr)
    elif not isinstance(arr, np.ndarray):
        print(f"Can't convert {type(arr)} to np.array")
    if arr.dtype == 'O': arr = np.array(arr, dtype=np.float32)
    return arr


# Cell
def decompress_from_url(url, target_dir=None, verbose=False):
    """Downloads a compressed file from its URL and uncompresses it.

    Parameters
    ----------
    url : string
        URL from which to download.
    target_dir : str or None (default: None)
        Directory to be used to extract downloaded files.
    verbose : bool (default: False)
        Whether to print information about the process (cached files used, ...)

    Returns
    -------
    str or None
        Directory in which the compressed file has been extracted if the process was
        successful, None otherwise
    """
    try:
        fname = os.path.basename(url)
        tmpdir = tempfile.mkdtemp()
        local_comp_fname = os.path.join(tmpdir, fname)
        urlretrieve(url, local_comp_fname)
    except:
        shutil.rmtree(tmpdir)
        if verbose:
            sys.stderr.write("Could not download url. Please, check url.\n")
    try:
        if not os.path.exists(target_dir): os.makedirs(target_dir)
        Archive(local_comp_fname).extractall(target_dir)
        shutil.rmtree(tmpdir)
        if verbose:
            print("Successfully extracted file %s to path %s" %
                  (local_comp_fname, target_dir))
        return target_dir
    except:
        shutil.rmtree(tmpdir)
        if verbose:
            sys.stderr.write("Could not uncompress file, aborting.\n")
        return None



def get_UCR_univariate_list():
    return sorted([
        'ACSF1', 'Adiac', 'AllGestureWiimoteX', 'AllGestureWiimoteY',
        'AllGestureWiimoteZ', 'ArrowHead', 'AsphaltObstacles', 'BME', 'Beef',
        'BeetleFly', 'BirdChicken', 'CBF', 'Car', 'Chinatown',
        'ChlorineConcentration', 'CinCECGTorso', 'Coffee', 'Computers',
        'CricketX', 'CricketY', 'CricketZ', 'Crop', 'DiatomSizeReduction',
        'DistalPhalanxOutlineAgeGroup', 'DistalPhalanxOutlineCorrect',
        'DistalPhalanxTW', 'DodgerLoopDay', 'DodgerLoopGame',
        'DodgerLoopWeekend', 'ECG200', 'ECG5000', 'ECGFiveDays',
        'EOGHorizontalSignal', 'EOGVerticalSignal', 'Earthquakes',
        'ElectricDevices', 'EthanolLevel', 'FaceAll', 'FaceFour', 'FacesUCR',
        'FiftyWords', 'Fish', 'FordA', 'FordB', 'FreezerRegularTrain',
        'FreezerSmallTrain', 'Fungi', 'GestureMidAirD1', 'GestureMidAirD2',
        'GestureMidAirD3', 'GesturePebbleZ1', 'GesturePebbleZ2', 'GunPoint',
        'GunPointAgeSpan', 'GunPointMaleVersusFemale',
        'GunPointOldVersusYoung', 'Ham', 'HandOutlines', 'Haptics', 'Herring',
        'HouseTwenty', 'InlineSkate', 'InsectEPGRegularTrain',
        'InsectEPGSmallTrain', 'InsectWingbeatSound', 'ItalyPowerDemand',
        'LargeKitchenAppliances', 'Lightning2', 'Lightning7', 'Mallat', 'Meat',
        'MedicalImages', 'MelbournePedestrian', 'MiddlePhalanxOutlineAgeGroup',
        'MiddlePhalanxOutlineCorrect', 'MiddlePhalanxTW',
        'MixedShapesRegularTrain', 'MixedShapesSmallTrain', 'MoteStrain',
        'NonInvasiveFetalECGThorax1', 'NonInvasiveFetalECGThorax2', 'OSULeaf',
        'OliveOil', 'PLAID', 'PhalangesOutlinesCorrect', 'Phoneme',
        'PickupGestureWiimoteZ', 'PigAirwayPressure', 'PigArtPressure',
        'PigCVP', 'Plane', 'PowerCons', 'ProximalPhalanxOutlineAgeGroup',
        'ProximalPhalanxOutlineCorrect', 'ProximalPhalanxTW',
        'RefrigerationDevices', 'Rock', 'ScreenType', 'SemgHandGenderCh2',
        'SemgHandMovementCh2', 'SemgHandSubjectCh2', 'ShakeGestureWiimoteZ',
        'ShapeletSim', 'ShapesAll', 'SmallKitchenAppliances', 'SmoothSubspace',
        'SonyAIBORobotSurface1', 'SonyAIBORobotSurface2', 'StarLightCurves',
        'Strawberry', 'SwedishLeaf', 'Symbols', 'SyntheticControl',
        'ToeSegmentation1', 'ToeSegmentation2', 'Trace', 'TwoLeadECG',
        'TwoPatterns', 'UMD', 'UWaveGestureLibraryAll', 'UWaveGestureLibraryX',
        'UWaveGestureLibraryY', 'UWaveGestureLibraryZ', 'Wafer', 'Wine',
        'WordSynonyms', 'Worms', 'WormsTwoClass', 'Yoga'
    ])


def get_UCR_multivariate_list():
    return sorted([
        'ArticularyWordRecognition', 'AtrialFibrillation', 'BasicMotions',
        'CharacterTrajectories', 'Cricket', 'DuckDuckGeese', 'ERing',
        'EigenWorms', 'Epilepsy', 'EthanolConcentration', 'FaceDetection',
        'FingerMovements', 'HandMovementDirection', 'Handwriting', 'Heartbeat',
        'InsectWingbeat', 'JapaneseVowels', 'LSST', 'Libras', 'MotorImagery',
        'NATOPS', 'PEMS-SF', 'PenDigits', 'PhonemeSpectra', 'RacketSports',
        'SelfRegulationSCP1', 'SelfRegulationSCP2', 'SpokenArabicDigits',
        'StandWalkJump', 'UWaveGestureLibrary'
    ])


# Cell
def get_UCR_univariate(sel_dataset, parent_dir='data/UCR', verbose=False, drop_na=False, check=True):
    if check and sel_dataset not in get_UCR_univariate_list():
        print('This dataset does not exist. Please select one from this list:')
        print(get_UCR_univariate_list())
        return None, None, None, None
    if verbose: print('Dataset:', sel_dataset)
    src_website = 'http://www.timeseriesclassification.com/Downloads/'
    tgt_dir = Path(parent_dir) / sel_dataset
    if verbose: print('Downloading and decompressing data...')
    if not os.path.isdir(tgt_dir):
        decompress_from_url(
            src_website + sel_dataset + '.zip', target_dir=tgt_dir, verbose=verbose)
    if verbose: print('...data downloaded and decompressed')
    fname_train = sel_dataset + "_TRAIN.arff"
    fname_test = sel_dataset + "_TEST.arff"

    train_df = pd.DataFrame(arff.loadarff(os.path.join(tgt_dir, fname_train))[0])
    test_df = pd.DataFrame(arff.loadarff(os.path.join(tgt_dir, fname_test))[0])
    unique_cats = train_df.iloc[:, -1].unique()
    mapping = dict(zip(unique_cats, np.arange(len(unique_cats))))
    train_df = train_df.replace({train_df.columns.values[-1]: mapping})
    test_df = test_df.replace({test_df.columns.values[-1]: mapping})
    if drop_na:
        train_df.dropna(axis=1, inplace=True)
        test_df.dropna(axis=1, inplace=True)

    X_train = train_df.iloc[:, :-1].values.astype(np.float32)
    X_test = test_df.iloc[:, :-1].values.astype(np.float32)
    y_train = train_df.iloc[:, -1].values.astype(int)
    y_test = test_df.iloc[:, -1].values.astype(int)

    X_train = To3dArray(X_train)
    X_test = To3dArray(X_test)

    if verbose:
        print('Successfully extracted dataset\n')
        print('X_train:', X_train.shape)
        print('y_train:', y_train.shape)
        print('X_valid:', X_test.shape)
        print('y_valid:', y_test.shape, '\n')
    return X_train, y_train, X_test, y_test



def get_UCR_multivariate(sel_dataset, parent_dir='data/UCR', verbose=False, check=True):
    if sel_dataset.lower() == 'mphoneme': sel_dataset = 'Phoneme'
    if check and sel_dataset not in get_UCR_multivariate_list():
        print('This dataset does not exist. Please select one from this list:')
        print(get_UCR_multivariate_list())
        return None, None, None, None
    if verbose: print('Dataset:', sel_dataset)
    src_website = 'http://www.timeseriesclassification.com/Downloads/'
    tgt_dir = Path(parent_dir) / sel_dataset

    if verbose: print('Downloading and decompressing data...')
    if not os.path.isdir(tgt_dir):
        decompress_from_url(
            src_website + sel_dataset + '.zip', target_dir=tgt_dir, verbose=verbose)
    if verbose: print('...data downloaded and decompressed')
    if verbose: print('Extracting data...')
    X_train_ = []
    X_test_ = []
    for i in range(10000):
        if not os.path.isfile(
                f'{parent_dir}/{sel_dataset}/{sel_dataset}Dimension'
                + str(i + 1) + '_TRAIN.arff'):
            break
        train_df = pd.DataFrame(
            arff.loadarff(
                f'{parent_dir}/{sel_dataset}/{sel_dataset}Dimension'
                + str(i + 1) + '_TRAIN.arff')[0])
        unique_cats = train_df.iloc[:, -1].unique()
        mapping = dict(zip(unique_cats, np.arange(len(unique_cats))))
        train_df = train_df.replace({train_df.columns.values[-1]: mapping})
        test_df = pd.DataFrame(
            arff.loadarff(
                f'{parent_dir}/{sel_dataset}/{sel_dataset}Dimension'
                + str(i + 1) + '_TEST.arff')[0])
        test_df = test_df.replace({test_df.columns.values[-1]: mapping})
        X_train_.append(train_df.iloc[:, :-1].values)
        X_test_.append(test_df.iloc[:, :-1].values)

    if verbose: print('...extraction complete')
    X_train = np.stack(X_train_, axis=-1)
    X_test = np.stack(X_test_, axis=-1)

    # In this case we need to rearrange the arrays ()
    X_train = np.transpose(X_train, (0, 2, 1))
    X_test = np.transpose(X_test, (0, 2, 1))

    y_train = np.array([int(float(x)) for x in train_df.iloc[:, -1]])
    y_test = np.array([int(float(x)) for x in test_df.iloc[:, -1]])

    if verbose:
        print('Successfully extracted dataset\n')
        print('X_train:', X_train.shape)
        print('y_train:', y_train.shape)
        print('X_valid:', X_test.shape)
        print('y_valid:', y_test.shape, '\n')
    return X_train, y_train, X_test, y_test


def get_UCR_data(dsid, parent_dir='data/UCR', verbose=False, check=True):
    if dsid in get_UCR_univariate_list():
        return get_UCR_univariate(dsid, verbose=verbose, check=check)
    elif dsid in get_UCR_multivariate_list():
        return get_UCR_multivariate(dsid, verbose=verbose, check=check)
    else:
        print(f'This {dsid} dataset does not exist. Please select one from these lists:')
        print('\nunivariate datasets')
        print(get_UCR_univariate_list())
        print('\nmultivariate datasets')
        print(get_UCR_multivariate_list(), '\n')
        return None, None, None, None


# Cell
def ucr_to_items(dset):
    '''
    create items for DataBlock from a UCR dset
    '''
    x_train, y_train, x_test, y_test = dset
    n_train = x_train.shape[0]
    return list(zip(np.concatenate([x_train, x_test]), np.concatenate([y_train, y_test]))), n_train

# Cell
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
def _get_x(df, x_cols, dtype=np.float32):
    return np.stack([df[x_cols[i]].values for i in range(len(x_cols))], axis=1).astype(dtype)
def _get_y(df, dep):
    return df[dep].values

# Cell
## not for discrete columns
def _calc_stats(x, n_train, axis=(0,2)):
    return np.nanmean(x[:n_train], axis=axis), np.nanstd(x[:n_train], axis=axis),np.nanmedian(x[:n_train], axis=axis)


# Cell
def _fillna(x, values):
    if np.ndim(values)==0: np.nan_to_num(x, copy=False, nan=0)

    else:
        #print(x.shape, values.shape)
#         assert x.shape[0]==values.shape[0]
        for i,v in enumerate(values): np.nan_to_num(x[:,i,...], copy=False, nan=v)

# Cell
def _normalize(x, means, stds):
    assert x.shape[-2]==means.shape[0]
    for i,v in enumerate(means): x[:,i,:]

# Cell
def items_from_df(df, cols_c, cols_y, n_train, cols_d=None, tab_cols_c=None , stats=None):
    '''
    creates timeseries items from a dataframe

    parameters:
    df: input dataframe
    cols_c: list of lists of columns for continuous valued channels (one list for each channel)
    cols_d: (optional) list of lists of columns for discrete valued channels
    cols_y: (list or single value) target column(s)
    n_train: int, neeeds to be provided for calculating the stats that are necessary to fill missing values
    tab_cols_c: (list or single value) tabular continunous columns
    stats: tuple (means, stds, medians)

    return a list of (xc,(xd),y) tuples (one list element for each dataframe row)
    '''

    cols_x = [cols_c]+[cols for cols in [cols_d, tab_cols_c]]
    _types = [np.float32, np.int16, np.float32]
#     print(cols_x)
    xs=[]
    for cols,t in zip(cols_x, _types):
        if cols is not None:
            x=_get_x(df, cols, dtype=t)
            axis=(0,2) if is_listy(cols[0]) else (0)
            if stats is None:
                means, stds, medians =  _calc_stats(x, n_train, axis=axis)
            else:
                means, stds, medians = stats #medians will be None right now
                if isinstance(means, torch.Tensor):
                    stds.torch('cpu').numpy()
                if isinstance(stds, torch.Tensor):
                    stds.to('cpu').numpy()
                if isinstance(medians, torch.Tensor):
                    medians.to('cpu').numpy()
                means, stds = means.squeeze(), stds.squeeze()

            #print(means.squeeze())
            #print(_calc_stats(x, n_train, axis=axis)[0])
            _fillna(x, means)
            assert not np.isnan(x).any()
            xs.append(x)

    y =  _get_y(df, cols_y)
#     return xs,y
    return list(zip(*xs, y))

# Cell
def _apply_cats(voc, add, c):
    if not is_categorical_dtype(c):
        return pd.Categorical(c, categories=voc[c.name][add:]).codes+add
    return c.cat.codes+add #if is_categorical_dtype(c) else c.map(voc[c.name].o2i)

def cats_from_df(df, tab_cols_cat, n_train, add_na=True):
    '''
    extract category codes for categorical columns from df, create 'na'

    parameters:
    df: input dataframe
    tab_cols_cat: list of categorical column names
    n_train: categories taken from df.iloc[:n_train] applied to df.iloc[n_train:]
    '''
    cat_maps = {c:CategoryMap(df[c].iloc[:n_train], add_na=add_na) for c in tab_cols_cat}  ## setup
    return np.stack([partial(_apply_cats,cat_maps,1)(df[c]) for c in tab_cols_cat], axis=1), cat_maps
#     return np.stack([pd.Cate])
