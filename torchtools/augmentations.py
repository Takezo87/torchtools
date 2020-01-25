# AUTOGENERATED! DO NOT EDIT! File to edit: 20_augmentations.ipynb (unless otherwise specified).

__all__ = ['noise_from_random_curve', 'noise_from_normal', 'YWarp', 'YNormal']

# Cell
import random
from functools import partial
from scipy.interpolate import CubicSpline
from .data import *

# Cell
import numpy as np
import torch

from fastai2.torch_basics import *
from fastai2.data.all import *

from .data import *

# Cell
#oguiza implementation, only used for _magwarp
# def random_curve_generator(ts, magnitude=.1, order=4, noise=None):
#     '''
#     sample points from a gaussian with mean 1 and create a smooth cubic "random curve" from these points
#     '''
#     seq_len = ts.shape[-1]
#     x = np.linspace(-seq_len, 2 * seq_len - 1, 3 * (order - 1) + 1, dtype=int)
#     x2 = np.random.normal(loc=1.0, scale=magnitude, size=len(x))
#     f = CubicSpline(x, x2, axis=-1)
#     return f(np.arange(seq_len))

# Cell

def _create_random_curve(n_channels, seq_len, magnitude, order):
    '''
    create a random curve for each channel in the interval[0, seq_len-1] on order random points
    '''
    x = np.linspace(0, seq_len-1, order)
    y = np.random.normal(loc=1.0, scale=magnitude, size=(n_channels, len(x)))
    f = CubicSpline(x, y, axis=-1)
    return f

def noise_from_random_curve(ts, magnitude=.1, order=4):
    '''
    sample points from a gaussian with mean 1 and create a smooth cubic "random curve" from these points
    ts, needs to be 2D
    order: number of sample to create the random curve from
    '''
    n_channels, seq_len = ts.shape
    f = _create_random_curve(n_channels, seq_len, magnitude, order)
    return torch.tensor(f(np.arange(seq_len)), dtype=ts.dtype, device=ts.device)

def noise_from_normal(ts, magnitude=.1):
    '''
    sample random noise from a gaussian with mean=1.0 and std=magnitude
    '''
    n_channels, seq_len = ts.shape
    return torch.tensor(np.random.normal(loc=1.0, scale=magnitude, size=(n_channels, seq_len)),
                        dtype=ts.dtype, device=ts.device)

# Cell
def _ynoise(x, magnitude=.1, add=True, smooth=True, **kwargs):
    '''
    add random noise to timeseries values
    '''
#     assert isinstance(x, Tensor)
    if magnitude <= 0: return x
    n_channels, seq_len = x.shape
    noise_fn = noise_from_random_curve if smooth else noise_from_normal

    noise = noise_fn(x, magnitude=magnitude, **kwargs)
    if add:
        output = x + (noise-1)
        return output
    else:
        output = x * (noise)
        return output

# Cell
_ynoise_warp = partial(_ynoise, smooth=True)
_ynoise_normal = partial(_ynoise, smooth=False)

# Cell
def _timenoise(x, magnitude=.1, smooth=False, **kwargs):
    '''This is a slow batch tfm on cpu'''
    if magnitude <= 0: return x
#     if len(x.shape)==1: x=x.unsqueeze(0) #no support for 1D tensors
    assert len(x.shape)==2, 'only 2D tensors supported'
    n_channels, seq_len = x.shape
    x_device = x.device ## make sure to put outpout on right device
    x=x.cpu() ## only works on cpu

#    return f
#     plt.plot(x.T)
#     plt.plot(np.linspace(0,10), f(np.linspace(0,10)[:, None]).squeeze())
    new_x = distort_time(x, magnitude=magnitude, smooth=True, **kwargs)
    fs = [CubicSpline(np.arange(seq_len), xi, axis=-1) for xi in x]
#     new_y = f(new_x, )
    new_y = torch.stack([torch.tensor(fs[i](xi)) for i,xi in enumerate(new_x)] )
    return new_y.to(x_device)

# Cell
class YWarp(Transform):
    order=200
    def encodes(self, x:TSTensor):
        print('timewarp')
        return _ynoise_warp(x)

# Cell
class YNormal(Transform):
    order=200
    def encodes(self, x:TSTensor):
        print('timenormal')
        return _ynoise_normal(x)