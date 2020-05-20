# AUTOGENERATED! DO NOT EDIT! File to edit: 40_datablock.ipynb (unless otherwise specified).

__all__ = ['TensorFloat', 'FixedSplitter', 'get_discrete_config']

# Cell
from .data import *
from .datasets import *

from .augmentations import *

from .models import *
from .core import *

# Cell
from fastai2.data.all import *
from fastai2.basics import *

# Cell
import pandas as pd
import numpy as np
from functools import partial

# Cell
#added squeeze
#should be in data or core
class TensorFloat(TensorBase):
    '''
    float target value of a timeseries
    ctx expected to be a `axes` object
    '''
    _show_args={}
#     def show(self, ctx=None, **kwargs):
#         if 'figsize' in kwargs: del kwargs['figsize']
#         ctx.suptitle(f'Label: {self.numpy()}', fontsize=16) ## ctx fig
#         return ctx

    def show(self, ctx=None, **kwargs):
        if 'figsize' in kwargs: del kwargs['figsize']
        assert ctx is not None; 'cannot show a label without ctx'
        ctx.set_title(f'Label: {self.squeeze().numpy():.2f}', fontsize=16) ## ctx axes object
        return ctx

# Cell
def FixedSplitter(end_train=10000, end_valid=15000):
    def _inner(o, **kwargs):
        return L(range(0, end_train)), L(range(end_train, end_valid))
    return _inner

# Cell
def get_discrete_config():
    '''get a simple column configuration for development'''
    x_cols_cont = [[f'x{i}_{j}' for j in range(10)] for i in [0,1,3,4]]
    x_cols_discrete = [[f'x{i}_{j}' for j in range(10)] for i in [2,5]]
    dep = 'y0'
    n_train = 8000

    return x_cols_cont, x_cols_discrete, dep, n_train