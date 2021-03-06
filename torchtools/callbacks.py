

from fastai.callback.all import *
from fastcore.basics import *
from fastcore.foundation import *
from fastai.losses import *
from fastai.imports import pv
from fastai.torch_core import Module

import torch
import torch.nn as nn
import math
import itertools
import numpy as np
from numpy import random
from pathlib import Path
import warnings
import os
import sklearn
from typing import Optional

from torchtools.core import *
import matplotlib.pyplot as plt

####### from tsai.utiils
def random_shuffle(o, random_state=None):
    res = sklearn.utils.shuffle(o, random_state=random_state)
    if isinstance(o, L): return L(list(res))
    return res

def cls_name(o): return o.__class__.__name__
#########################33

def create_subsequence_mask(o, r=.15, lm=3, stateful=True, sync=False):
    if o.ndim == 2: o = o[None]
    n_masks, mask_dims, mask_len = o.shape
    if sync == 'random': sync = random.random() > .5
    dims = 1 if sync else mask_dims
    numels = n_masks * dims * mask_len
    pm = 1 / lm
    pu = pm * (r / (1 - r))
    a, b, proba_a, proba_b = ([1], [0], pu, pm) if random.random() > pm else ([0], [1], pm, pu)
    if stateful:
        max_len = max(1, 2 * math.ceil(numels // (1/pm + 1/pu)))
        while True:
            dist_a = np.clip(np.random.geometric(proba_a, max_len), 1, mask_len)
            dist_b = np.clip(np.random.geometric(proba_b, max_len), 1, mask_len)
            if (dist_a + dist_b).sum() >= numels:
                dist_len = np.argmax((dist_a + dist_b).cumsum() >= numels) + 1
                break
        l = [a*ax + b*bx for (ax, bx) in zip(dist_a[:dist_len], dist_b[:dist_len])]
        _mask = list(itertools.chain.from_iterable(l))[:numels]
    else:
        _mask = np.random.binomial(1, 1 - r, numels)
    mask = torch.Tensor(_mask).reshape(n_masks, dims, mask_len)
    if sync: mask = mask.repeat(1, mask_dims, 1)
    return mask.to(o.device)

def create_variable_mask(o, r=.15):
    n_masks, mask_dims, mask_len = o.shape
    sel_dims = np.random.choice(n_masks * mask_dims, int(n_masks * mask_dims * r), False)
    _mask = np.ones((n_masks * mask_dims, mask_len))
    _mask[sel_dims] = 0
    mask = torch.Tensor(_mask).reshape(*o.shape)
    return mask.to(o.device)

def create_future_mask(o, r=.15):
    n_masks, mask_dims, mask_len = o.shape
    sel_steps = int(round(mask_len * r))
    _mask = np.ones((1, 1, mask_len))
    _mask[..., -sel_steps:] = 0
    mask = torch.Tensor(_mask).repeat(n_masks, mask_dims, 1)
    return mask.to(o.device)

# Cell
class TSBERT_Loss(Module):
    def __init__(self, crit=None):
        self.crit = ifnone(crit, MSELossFlat())
        self.mask = slice(None)

    def forward(self, preds, target):
        return self.crit(preds[self.mask], target[self.mask])

# Cell
import matplotlib.colors as mcolors


class TSBERT(Callback):
    def __init__(self, r: float = .15, subsequence_mask: bool = True, lm: float = 3., stateful: bool = True, sync: bool = False, variable_mask: bool = False,
                 future_mask: bool = False, custom_mask: Optional = None, dropout: float = .1, crit: callable = None,
                 target_dir: str = './data/TSBERT', fname: str = 'model', verbose: bool = True):
        r"""
        Callback used to perform the autoregressive task of denoising the input after a binary mask has been applied.
        Args:
            r: proba of masking.
            subsequence_mask: apply a mask to random subsequences.
            lm: average mask len when using stateful (geometric) masking.
            stateful: geometric distribution is applied so that average mask length is lm.
            sync: all variables have the same masking.
            variable_mask: apply a mask to random variables.
            future_mask: used to train a forecasting model.
            custom_mask: allows to pass any type of mask with input tensor and output tensor.
            dropout: dropout applied to the head of the model during pretraining.
            crit: loss function that will be used. If None MSELossFlat().
            target_dir : directory where trained model will be stored.
            fname : file name that will be used to save the pretrained model.
    """
        assert subsequence_mask or variable_mask or future_mask or custom_mask, \
            'you must set (subsequence_mask and/or variable_mask) or future_mask to True or use a custom_mask'
        if custom_mask is not None and (future_mask or subsequence_mask or variable_mask):
            warnings.warn("Only custom_mask will be used")
        elif future_mask and (subsequence_mask or variable_mask):
            warnings.warn("Only future_mask will be used")
        store_attr(
            "subsequence_mask,variable_mask,future_mask,custom_mask,dropout,r,lm,stateful,sync,crit,fname,verbose")
        self.target_dir = Path(target_dir)

    def before_fit(self):
        # modify loss for denoising task
        self.old_loss_func = self.learn.loss_func
        self.learn.loss_func = TSBERT_Loss(self.crit)
        self.learn.TSBERT = self

        # remove and store metrics
        self.learn.metrics = L([])

        # change head with conv layer (equivalent to linear layer applied to dim=1)
        self.learn.model.head = nn.Sequential(nn.Dropout(self.dropout), nn.Conv1d(
            self.learn.model.head_nf, self.learn.dls.vars, 1)).to(self.learn.dls.device)

    def before_batch(self):
        if self.custom_mask is not None:
            mask = self.custom_mask(self.x)
        elif self.future_mask:
            mask = create_future_mask(self.x, r=self.r)
        elif self.subsequence_mask and self.variable_mask:
            random_thr = 1/3 if self.sync == 'random' else 1/2
            if random.random() > random_thr:
                mask = create_subsequence_mask(
                    self.x, r=self.r, lm=self.lm, stateful=self.stateful, sync=self.sync)
            else:
                mask = create_variable_mask(self.x, r=self.r)
        elif self.subsequence_mask:
            mask = create_subsequence_mask(
                self.x, r=self.r, lm=self.lm, stateful=self.stateful, sync=self.sync)
        elif self.variable_mask:
            mask = create_variable_mask(self.x, r=self.r)
        else:
            raise ValueError(
                'You need to set subsequence_mask and/ or variable_mask to True in TSBERT.')

        self.learn.yb = (self.x,)
        self.learn.xb = (self.x * mask,)
        self.learn.loss_func.mask = (mask == 0)  # boolean mask
        self.mask = mask

    def after_fit(self):
        if self.epoch == self.n_epoch - 1 and not "LRFinder" in [cls_name(cb) for cb in self.learn.cbs]:
            PATH = Path(f'{self.target_dir/self.fname}.pth')
            if not os.path.exists(PATH.parent):
                os.makedirs(PATH.parent)
            torch.save(self.learn.model.state_dict(), PATH)
            pv(f"\npre-trained model weights_path='{PATH}'\n", self.verbose)

    def show_preds(self, max_n=9, nrows=3, ncols=3, figsize=None, sharex=True, **kwargs):
        b = self.learn.dls.valid.one_batch()
        self.learn._split(b)
        xb = self.xb[0].detach().cpu().numpy()
        bs, nvars, seq_len = xb.shape
        self.learn('before_batch')
        pred = self.learn.model(*self.learn.xb).detach().cpu().numpy()
        mask = self.mask.cpu().numpy()
        masked_pred = np.ma.masked_where(mask, pred)
        ncols = min(ncols, math.ceil(bs / ncols))
        nrows = min(nrows, math.ceil(bs / ncols))
        max_n = min(max_n, bs, nrows*ncols)
        if figsize is None:
            figsize = (ncols*6, math.ceil(max_n/ncols)*4)
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols,
                               figsize=figsize, sharex=sharex, **kwargs)
        idxs = np.random.permutation(np.arange(bs))
        colors = list(mcolors.TABLEAU_COLORS.keys()) + \
            random_shuffle(list(mcolors.CSS4_COLORS.keys()))
        i = 0
        for row in ax:
            for col in row:
                color_iter = iter(colors)
                for j in range(nvars):
                    try:
                        color = next(color_iter)
                    except:
                        color_iter = iter(colors)
                        color = next(color_iter)
                    col.plot(xb[idxs[i]][j], alpha=.5, color=color)
                    col.plot(masked_pred[idxs[i]][j],
                             marker='o', markersize=4, color=color)
                i += 1
        plt.tight_layout()
        plt.show()
