# AUTOGENERATED! DO NOT EDIT! File to edit: 60_dataloader.ipynb (unless otherwise specified).

__all__ = ['cpus', 'device', 'bytes2GB', 'totensor', 'toarray', 'to3dtensor', 'to2dtensor', 'to1dtensor', 'to3darray',
           'to2darray', 'to1darray', 'to3d', 'to2d', 'to1d', 'to2dPlus', 'to3dPlus', 'to2dPlusTensor', 'to2dPlusArray',
           'to3dPlusTensor', 'to3dPlusArray', 'Todtype', 'itemify', 'ifnoneelse', 'cycle_dl', 'stack', 'NumpyTensor',
           'NumpyDatasets', 'TSDatasets3', 'TSDatasets4', 'TSDatasets5', 'NumpyTensorBlock', 'TSTensorBlock',
           'NumpyDataLoader', 'show_tuple', 'TSDataLoader', 'NumpyDataLoaders', 'TSDataLoaders', 'TSStandardize',
           'TSNormalize', 'items_to_arrays']

# Cell
import numpy as np
import scipy as sp
#import torch
from fastai2.torch_basics import *
from fastai2.data.all import *
from fastai2.callback.all import *

from fastai2.data.all import *
from fastai2.basics import *

# Cell
import psutil
import fastai2
import fastcore
import torch

# Cell
from .data import *
from .datasets import *
# from torchtools.augmentations import *
from .datablock import *

# Cell
from .models import *
from .core import *

import pandas as pd
import numpy as np
from functools import partial

_verbose=True

# Cell
#tsai.imports
cpus = defaults.cpus
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Cell
#tsai.utils
def bytes2GB(byts):
    return round(byts / math.pow(1024, 3), 2)

# Cell
#tsai.utils
#export
def totensor(o):
    if isinstance(o, torch.Tensor): return o
    elif isinstance(o, np.ndarray):  return torch.from_numpy(o)
    assert False, f"Can't convert {type(o)} to torch.Tensor"


def toarray(o):
    if isinstance(o, np.ndarray): return o
    elif isinstance(o, torch.Tensor): return o.cpu().numpy()
    assert False, f"Can't convert {type(o)} to np.array"


def to3dtensor(o):
    o = totensor(o)
    if o.ndim == 3: return o
    elif o.ndim == 1: return o[None, None]
    elif o.ndim == 2: return o[:, None]
    assert False, f'Please, review input dimensions {o.ndim}'


def to2dtensor(o):
    o = totensor(o)
    if o.ndim == 2: return o
    elif o.ndim == 1: return o[None]
    elif o.ndim == 3: return o[0]
    assert False, f'Please, review input dimensions {o.ndim}'


def to1dtensor(o):
    o = totensor(o)
    if o.ndim == 1: return o
    elif o.ndim == 3: return o[0,0]
    if o.ndim == 2: return o[0]
    assert False, f'Please, review input dimensions {o.ndim}'


def to3darray(o):
    o = toarray(o)
    if o.ndim == 3: return o
    elif o.ndim == 1: return o[None, None]
    elif o.ndim == 2: return o[:, None]
    assert False, f'Please, review input dimensions {o.ndim}'


def to2darray(o):
    o = toarray(o)
    if o.ndim == 2: return o
    elif o.ndim == 1: return o[None]
    elif o.ndim == 3: return o[0]
    assert False, f'Please, review input dimensions {o.ndim}'


def to1darray(o):
    o = toarray(o)
    if o.ndim == 1: return o
    elif o.ndim == 3: o = o[0,0]
    elif o.ndim == 2: o = o[0]
    assert False, f'Please, review input dimensions {o.ndim}'


def to3d(o):
    if o.ndim == 3: return o
    if isinstance(o, np.ndarray): return to3darray(o)
    if isinstance(o, torch.Tensor): return to3dtensor(o)


def to2d(o):
    if o.ndim == 2: return o
    if isinstance(o, np.ndarray): return to2darray(o)
    if isinstance(o, torch.Tensor): return to2dtensor(o)


def to1d(o):
    if o.ndim == 1: return o
    if isinstance(o, np.ndarray): return to1darray(o)
    if isinstance(o, torch.Tensor): return to1dtensor(o)


def to2dPlus(o):
    if o.ndim >= 2: return o
    if isinstance(o, np.ndarray): return to2darray(o)
    elif isinstance(o, torch.Tensor): return to2dtensor(o)


def to3dPlus(o):
    if o.ndim >= 3: return o
    if isinstance(o, np.ndarray): return to3darray(o)
    elif isinstance(o, torch.Tensor): return to3dtensor(o)


def to2dPlusTensor(o):
    return to2dPlus(totensor(o))


def to2dPlusArray(o):
    return to2dPlus(toarray(o))


def to3dPlusTensor(o):
    return to3dPlus(totensor(o))


def to3dPlusArray(o):
    return to3dPlus(toarray(o))


def Todtype(dtype):
    def _to_type(o, dtype=dtype):
        if o.dtype == dtype: return o
        elif isinstance(o, torch.Tensor): o = o.to(dtype=dtype)
        elif isinstance(o, np.ndarray): o = o.astype(dtype)
        return o
    return _to_type

# Cell
#tsai.utils
#export
def itemify(*o, tup_id=None):
    items = L(*o).zip()
    if tup_id is not None: return L([item[tup_id] for item in items])
    else: return items

def ifnoneelse(a, b, c=None):
    "`b` if `a` is None else `c`"
    return b if a is None else ifnone(c, a)

def cycle_dl(dl):
    for _ in dl: _

# Cell
#fastcore.foundations
def _is_array(x): return hasattr(x,'__array__') or hasattr(x,'iloc')
def _listify(o):
    if o is None: return []
    if isinstance(o, list): return o
    if isinstance(o, str) or _is_array(o): return [o]
    if is_iter(o): return list(o)
    return [o]

# Cell
#tsai.utils
def stack(o, axis=0):
    if isinstance(o[0], torch.Tensor): return torch.stack(tuple(o), dim=axis)
    else: return np.stack(o, axis)

# Cell
#tsai.data.core
class NumpyTensor(TensorBase):
    "Returns a `tensor` with subclass `NumpyTensor` that has a show method"
    def __new__(cls, o, **kwargs):
        if isinstance(o, (list, L)): o = stack(o)
        res = cast(tensor(o), cls)
        res._meta = kwargs
        return res
    def __getitem__(self, idx):
        res = super().__getitem__(idx)
        return res.as_subclass(type(self))
    def __repr__(self):
        if self.numel() == 1: return f'{self}'
        else: return f'NumpyTensor(shape:{list(self.shape)})'
    def show(self, ax=None, ctx=None, title=None, title_color='black', **kwargs):
        if self.ndim != 2: self = type(self)(to2dtensor(self))
        ax = ifnone(ax,ctx)
        if ax is None: fig, ax = plt.subplots(**kwargs)
        ax.plot(self.T)
        ax.axis(xmin=0, xmax=self.shape[-1] - 1)
        ax.set_title(title, weight='bold', color=title_color)
        plt.tight_layout()
        return ax

# Cell
#tsai.data.core
class NumpyDatasets(Datasets):
    "A dataset that creates tuples from X (and y) and applies `tfms` of type item_tfms"
    _xtype, _ytype = NumpyTensor, None # Expected X and y output types (must have a show method)
    def __init__(self, X=None, y=None, items=None, tfms=None, tls=None, n_inp=None, dl_type=None, inplace=True, **kwargs):
        self.inplace = inplace
        if tls is None:
            X = itemify(X, tup_id=0)
            y = itemify(y, tup_id=0) if y is not None else y
            items = tuple((X,)) if y is None else tuple((X,y))
            self.tfms = L(ifnone(tfms,[None]*len(ifnone(tls,items))))
        self.tls = L(tls if tls else [TfmdLists(item, t, **kwargs) for item,t in zip(items,self.tfms)])
        self.n_inp = (1 if len(self.tls)==1 else len(self.tls)-1) if n_inp is None else n_inp
        if len(self.tls[0]) > 0:
            self.types = L([ifnone(_typ, type(tl[0]) if isinstance(tl[0], torch.Tensor) else tensor)
                            for tl,_typ in zip(self.tls, [self._xtype, self._ytype])])
            self.ptls = L([tl if not self.inplace else tl[:] if type(tl[0]).__name__ == 'memmap'
                           else tensor(stack(tl[:])) for tl in self.tls])

    def __getitem__(self, it):
        return tuple([typ(ptl[it]) for i,(ptl,typ) in enumerate(zip(self.ptls,self.types))])

    def subset(self, i): return type(self)(tls=L(tl.subset(i) for tl in self.tls), n_inp=self.n_inp, inplace=self.inplace, tfms=self.tfms)

    def _new(self, X, *args, y=None, **kwargs):
        items = ifnoneelse(y,tuple((X,)),tuple((X, y)))
        return super()._new(items, tfms=self.tfms, do_setup=False, **kwargs)

    def show_at(self, idx, **kwargs):
        self.show(self[idx], **kwargs)
        plt.show()

    @property
    def items(self): return tuple([tl.items for tl in self.tls])
    @items.setter
    def items(self, vs):
        for tl,c in zip(self.tls, vs): tl.items = v

# Cell
#tsai.data.core
## slightly adapted version
##NOTE TODO: Why does _ytype=TensorFloat not work (autograd fails)
class TSDatasets3(NumpyDatasets):
    "A dataset that creates tuples from X (and y) and applies `item_tfms`"
    _xtype, _xdistype, _ytype = TSTensor, TSIntTensor, None # Expected X and y output types (torch.Tensor - default - or subclass)
    def __init__(self, X=None, X_dis=None, y=None, items=None, sel_vars=None, sel_steps=None, tfms=None, tls=None, n_inp=None, dl_type=None,
                 inplace=True, **kwargs):
        self.inplace = inplace

        if tls is None:
            X = itemify(to3darray(X), tup_id=0)
            X_dis = itemify(to3darray(X_dis), tup_id=0) if X_dis is not None else X_dis
            #toarray(y) only needed if y-elements are not scalars, toarray is time consuming
            y = itemify(toarray(y), tup_id=0) if y is not None else y
            items = tuple((X,)) if y is None else tuple((X,y))
            if X_dis is not None: items = tuple((X, X_dis, y)) if y is not None else tuple(X, X_dis,)
            self.tfms = L(ifnone(tfms,[None]*len(ifnone(tls,items))))

#         if X_dis is not None: self.X_dis = X_dis

        self.sel_vars = ifnone(sel_vars, slice(None))
        self.sel_steps = ifnone(sel_steps,slice(None))
#         self.splits_help = splits
        self.tls = L(tls if tls else [TfmdLists(item, t, **kwargs) for item,t in zip(items,self.tfms)])
        self.n_inp = (1 if len(self.tls)==1 else len(self.tls)-1) if n_inp is None else n_inp
        if len(self.tls[0]) > 0:
            _tls_types = [self._xtype, self._ytype] if len(self.tls)==2 else [self._xtype, self._xdistype, self._ytype]
#             print(_tls_types)
#             print(len(self.tls))
#             for tl,_typ in zip(self.tls, _tls_types):
#                 print (len(tl), _typ, type(tl[0]), isinstance(tl[0], torch.Tensor))
            self.types = L([ifnone(_typ, type(tl[0]) if isinstance(tl[0], torch.Tensor) else tensor) for
                            tl,_typ in zip(self.tls, _tls_types)])

            self.types = L([ifnone(_typ, type(tl[0]) if isinstance(tl[0], torch.nn.Sequential) else tensor) for
                            tl,_typ in zip(self.tls, _tls_types)])
            if self.inplace and X and y: self.ptls=L(
                [tensor(X), tensor(y)]) if not X_dis else L([tensor(X), tensor(X_dis), tensor(y)])
            else:
                self.ptls = L([tl if not self.inplace else tl[:] if type(tl[0]).__name__ == 'memmap' else
                               tensor(stack(tl[:])) for tl in self.tls])

    def __getitem__(self, it):

#         for i,(ptl,typ) in enumerate(zip(self.ptls,self.types)):
#             print (i, typ)

#         return tuple([typ(ptl[it])[...,self.sel_vars, self.sel_steps] if i==0 else
#                       typ(ptl[it]) for i,(ptl,typ) in enumerate(zip(self.ptls,self.types))])
        ## do not enable slicing for now
        return tuple([typ(ptl[it]) for i,(ptl,typ) in enumerate(zip(self.ptls,self.types))])


    def subset(self, i):
        if self.inplace:
            X = self.ptls[0][self.splits[i]]
            y = self.ptls[-1][self.splits[i]]
            X_dis = None if len(self.ptls)==2 else self.ptls[1][self.splits[i]]
            #if X_dis:print(X.shape, y.shape, X_dis.shape)
            res = type(self)(X=X, X_dis=X_dis, y=y, n_inp=self.n_inp,
                                           inplace=self.inplace, tfms=self.tfms,
                                           sel_vars=self.sel_vars, sel_steps=self.sel_steps)
            res.set_split_idx_fixed(i)
            return res


        else:
            return type(self)(tls=L(tl.subset(i) for tl in self.tls), n_inp=self.n_inp,
                                           inplace=self.inplace, tfms=self.tfms,
                                           sel_vars=self.sel_vars, sel_steps=self.sel_steps)
    @property
    def vars(self): return self[0][0].shape[-2]
    @property
    def len(self): return self[0][0].shape[-1]

    ## do not confuse with set_split_idx contextmanager in fastai2 Datasets
    def set_split_idx_fixed(self, i):
        for tl in self.tls: tl.tfms.split_idx = i

# Cell
#tsai.data.core
## slightly adapted version
##NOTE TODO: Why does _ytype=TensorFloat not work (autograd fails)
class TSDatasets4(NumpyDatasets):
    "A dataset that creates tuples from X (and y) and applies `item_tfms`"
    _xtype, _xdistype, _xtabctype, _xtabcattype, _ytype = TSTensor, TSIntTensor, None, None, None # Expected X and y output types (torch.Tensor - default - or subclass)
    def __init__(self, X=None, X_dis=None, y=None, items=None, sel_vars=None, sel_steps=None, tfms=None, tls=None, n_inp=None, dl_type=None,
                 inplace=True, X_tabc=None, X_tabcat=None, **kwargs):
        self.inplace = inplace
        self.has_xtype=[X is not None, X_dis is not None, X_tabc is not None, X_tabcat is not None]

        if tls is None:
            X = itemify(to3darray(X), tup_id=0) if X is not None else X
            X_dis = itemify(to3darray(X_dis), tup_id=0) if X_dis is not None else X_dis
            X_tabc = itemify(toarray(X_tabc), tup_id=0) if X_tabc is not None else X_tabc
            X_tabcat = itemify(toarray(X_tabcat), tup_id=0) if X_tabcat is not None else X_tabcat
            #toarray(y) only needed if y-elements are not scalars, toarray is time consuming
            y = itemify(toarray(y), tup_id=0) if y is not None else y
            items = tuple((X,)) if y is None else tuple(x for x in [X,X_dis, X_tabc, X_tabcat, y] if x is not None)
#             if X_dis is not None: items = tuple((X, X_dis, y)) if y is not None else tuple(X, X_dis,)
            self.tfms = L(ifnone(tfms,[None]*len(ifnone(tls,items))))

#         if X_dis is not None: self.X_dis = X_dis

        self.sel_vars = ifnone(sel_vars, slice(None))
        self.sel_steps = ifnone(sel_steps,slice(None))
#         self.splits_help = splits
        self.tls = L(tls if tls else [TfmdLists(item, t, **kwargs) for item,t in zip(items,self.tfms)])
        self.n_inp = (1 if len(self.tls)==1 else len(self.tls)-1) if n_inp is None else n_inp
        if len(self.tls[0]) > 0:
#             print(_xtype)
            _tls_types=[t for x,t in zip(
                [X,X_dis, X_tabc, X_tabcat, y], [self._xtype, self._xdistype,
                                                 self._xtabctype, self._xtabcattype, self._ytype])
                        if x is not None]

            print(_tls_types)
            #_tls_types = [self._xtype, self._ytype] if len(self.tls)==2 else [self._xtype, self._xdistype, self._ytype]
#             print(_tls_types)
#             print(len(self.tls))
#             for tl,_typ in zip(self.tls, _tls_types):
#                 print (len(tl), _typ, type(tl[0]), isinstance(tl[0], torch.Tensor))
            self.types = L([ifnone(_typ, type(tl[0]) if isinstance(tl[0], torch.Tensor) else tensor) for
                            tl,_typ in zip(self.tls, _tls_types)])

#             self.types = L([ifnone(_typ, type(tl[0]) if isinstance(tl[0], torch.nn.Sequential) else tensor) for
#                             tl,_typ in zip(self.tls, _tls_types)])

            if self.inplace and X and y:

                self.ptls=L([tensor(x) for x in [X,X_dis, X_tabc, X_tabcat] if x is not None]+[tensor(y)])
#                 [tensor(X), tensor(y)]) if not X_dis else L([tensor(X), tensor(X_dis), tensor(y)])


#             if self.inplace and X and y: self.ptls=L(
#                 [tensor(X), tensor(y)]) if not X_dis else L([tensor(X), tensor(X_dis), tensor(y)])
            else:
                self.ptls = L([tl if not self.inplace else tl[:] if type(tl[0]).__name__ == 'memmap' else
                               tensor(stack(tl[:])) for tl in self.tls])

    def __getitem__(self, it):

#         for i,(ptl,typ) in enumerate(zip(self.ptls,self.types)):
#             print (i, typ)

#         return tuple([typ(ptl[it])[...,self.sel_vars, self.sel_steps] if i==0 else
#                       typ(ptl[it]) for i,(ptl,typ) in enumerate(zip(self.ptls,self.types))])
        ## do not enable slicing for now
        return tuple([typ(ptl[it]) for i,(ptl,typ) in enumerate(zip(self.ptls,self.types))])


    def subset(self, i):
        if self.inplace:
            X_type_idxs = [i for i in range(4) if self.has_xtype[i]]
            Xs = ['X', 'X_dis', 'X_tabc', 'X_tabcat']
            X_dict=defaultdict(lambda:None)
            for j,k in enumerate(X_type_idxs):
                X_dict[Xs[k]]=  self.ptls[j][self.splits[i]]

            X,X_dis,X_tabc,X_tabcat = map(X_dict.__getitem__, Xs)



#             X = None if self.has_xtype[0] is False else self.ptls[0][self.splits[i]]
#             X_dis = None if self.has_xtype[1] is False else self.ptls[1-self.has_xtype[0]][self.splits[i]]
#             X_tabc = None if self.has_xtype[2] is False else self.ptls[
#                 2-self.has_xtype[0]-self.has_xtype[1]][self.splits[i]]
#             X_tabcat = None if self.has_xtype[3] is False else self.ptls[
#                 3-self.has_xtype[0]-self.has_xtype[1]-self.has_xtype[2]][self.splits[i]]][self.splits[i]]
            y = self.ptls[-1][self.splits[i]]

            #if X_dis:print(X.shape, y.shape, X_dis.shape)
            res = type(self)(X=X, X_dis=X_dis, X_tabc=X_tabc, X_tabcat=X_tabcat, y=y, n_inp=self.n_inp,
                                           inplace=self.inplace, tfms=self.tfms,
                                           sel_vars=self.sel_vars, sel_steps=self.sel_steps)
            res.set_split_idx_fixed(i)
            return res


        else:
            return type(self)(tls=L(tl.subset(i) for tl in self.tls), n_inp=self.n_inp,
                                           inplace=self.inplace, tfms=self.tfms,
                                           sel_vars=self.sel_vars, sel_steps=self.sel_steps)
    @property
    def vars(self): return self[0][0].shape[-2]
    @property
    def len(self): return self[0][0].shape[-1]

    ## do not confuse with set_split_idx contextmanager in fastai2 Datasets
    def set_split_idx_fixed(self, i):
        for tl in self.tls: tl.tfms.split_idx = i

# Cell
#tsai.data.core
## adapted version

##Note: For this version of Datasets, item transforms are not propagated, transformed lists more or less pointless?? It is much faster though

class TSDatasets5(NumpyDatasets):
    "A dataset that creates tuples from X (and y) and applies `item_tfms`"
    _xctype, _xdtype, _xtconttype, _xtcattype, _ytype = TSTensor, TSIntTensor, None, None, None # Expected X and y output types (torch.Tensor - default - or subclass)
    def __init__(self, X_c=None, X_d=None, y=None, items=None, tfms=None, tls=None, n_inp=None, dl_type=None,
                 inplace=True, X_tcont=None, X_tcat=None, has_x=None, _ytype=None, **kwargs):
        self.inplace = inplace ## should be always True for this implementation
        self._ytype = _ytype
        self.has_x = ifnone(has_x, [X_c is not None, X_d is not None,
                                            X_tcont is not None, X_tcat is not None])

        if tls is None: ## always None in this implementation
            X_c = itemify(to3darray(X_c), tup_id=0) if X_c is not None else X_c
            X_d = itemify(to3darray(X_d), tup_id=0) if X_d is not None else X_d
            X_tcont = itemify(toarray(X_tcont), tup_id=0) if X_tcont is not None else X_tcont
            X_tcat = itemify(toarray(X_tcat), tup_id=0) if X_tcat is not None else X_tcat
            #toarray(y) only needed if y-elements are not scalars, toarray is time consuming
            y = itemify(toarray(y), tup_id=0) if y is not None else y
            items = tuple((X_c,)) if y is None else tuple(x for x in [X_c, X_d, X_tcont, X_tcat, y]
                                                          if x is not None)
            self.tfms = L(ifnone(tfms,[None]*len(ifnone(tls,items))))

        self.tls = L(tls if tls else [TfmdLists(item, t, **kwargs) for item,t in zip(items,self.tfms)])
        self.n_inp = (1 if len(self.tls)==1 else len(self.tls)-1) if n_inp is None else n_inp
        if len(self.tls[0]) > 0:
#             print(_xtype)
            _tls_types=L([self._xctype, self._xdtype, self._xtconttype, self._xtcattype])[self.has_x]+L([self._ytype])
#             _tls_types=[t for x,t in zip(self.has_x, [self._xctype, self._xdtype, self._xtconttype, self._xtcattype, self._ytype])
#                         if x]
            print(_tls_types)

            self.types = [ifnone(_typ, type(tl[0]) if isinstance(tl[0], torch.Tensor) else tensor) for
                            tl,_typ in zip(self.tls, _tls_types)]

            if self.inplace:
                print('fast part')
                self.ptls=L([tensor(x) for x in [X_c, X_d, X_tcont, X_tcat,y] if x is not None])
            else:
        #this part should never be called in this implementation, observe that the item transforms
        #in the original fastai2 datasets are applied by slicing into the TfmdLists
                print('slow part')
                self.ptls = L([tl if not self.inplace else tl[:] if type(tl[0]).__name__ == 'memmap' else
                               tensor(stack(tl[:])) for tl in self.tls])

    def __getitem__(self, it):
        return tuple([typ(ptl[it]) for i,(ptl,typ) in enumerate(zip(self.ptls,self.types))])

#     @property
    def subset(self, i):
#         return type(self)(tls=L(tl.subset(i) for tl in self.tls), n_inp=self.n_inp,
#                                            inplace=self.inplace, tfms=self.tfms,
#                                            sel_vars=self.sel_vars, sel_steps=self.sel_steps,
#                           has_xtype=self.has_xtype)
        if self.inplace:
            Xs = [x[self.splits[i]] for x in self.ptls[:-1]]
            X_c,X_d,X_tcont,X_tcat = map_xs(Xs, self.has_x)
            y = np.array(self.ptls[-1][self.splits[i]])

            #if X_dis:print(X.shape, y.shape, X_dis.shape)
            res = type(self)(X_c=X_c, X_d=X_d, X_tcont=X_tcont, X_tcat=X_tcat, y=y, n_inp=self.n_inp,
                                           inplace=self.inplace, tfms=self.tfms, _ytype=self._ytype)
            res.set_split_idx_fixed(i)
            return res


        else:
            return type(self)(tls=L(tl.subset(i) for tl in self.tls), n_inp=self.n_inp,
                                           inplace=self.inplace, tfms=self.tfms,
                                           sel_vars=self.sel_vars, sel_steps=self.sel_steps)
    @property
    def vars(self): return self[0][0].shape[-2]
    @property
    def len(self): return self[0][0].shape[-1]

    ## do not confuse with set_split_idx contextmanager in fastai2 Datasets
    def set_split_idx_fixed(self, i):
        for tl in self.tls: tl.tfms.split_idx = i

# Cell
#tsai.data.core

class NumpyTensorBlock():
    def __init__(self, type_tfms=None, item_tfms=None, batch_tfms=None, dl_type=None, dls_kwargs=None):
        self.type_tfms  =                 L(type_tfms)
        self.item_tfms  = ToNumpyTensor + L(item_tfms)
        self.batch_tfms =                 L(batch_tfms)
        self.dl_type,self.dls_kwargs = dl_type,({} if dls_kwargs is None else dls_kwargs)

class TSTensorBlock():
    def __init__(self, type_tfms=None, item_tfms=None, batch_tfms=None, dl_type=None, dls_kwargs=None):
        self.type_tfms  =              L(type_tfms)
        self.item_tfms  = ToTSTensor + L(item_tfms)
        self.batch_tfms =              L(batch_tfms)
        self.dl_type,self.dls_kwargs = dl_type,({} if dls_kwargs is None else dls_kwargs)

# Cell
#tsai.data.core
_batch_tfms = ('after_item','before_batch','after_batch')

class NumpyDataLoader(TfmdDL):
    idxs = None
    do_item = noops # create batch returns indices
    def __init__(self, dataset, bs=64, shuffle=False, num_workers=None, verbose=False, do_setup=True, batch_tfms=None, **kwargs):
        '''batch_tfms == after_batch (either can be used)'''
        if num_workers is None: num_workers = min(16, defaults.cpus)
        for nm in _batch_tfms:
            if nm == 'after_batch' and batch_tfms is not None: kwargs[nm] = Pipeline(batch_tfms)
            else: kwargs[nm] = Pipeline(kwargs.get(nm,None))
        bs = min(bs, len(dataset))
        super().__init__(dataset, bs=bs, shuffle=shuffle, num_workers=num_workers, **kwargs)
        if do_setup:
            for nm in _batch_tfms:
                pv(f"Setting up {nm}: {kwargs[nm]}", verbose)
                kwargs[nm].setup(self)

    def create_batch(self, b):
        it = b if self.shuffle else slice(b[0], b[0] + self.bs)
        self.idxs = b
        return self.dataset[it]

    def create_item(self, s): return s

    def get_idxs(self):
        idxs = Inf.count if self.indexed else Inf.nones
        if self.n is not None: idxs = list(range(len(self.dataset)))
        if self.shuffle: idxs = self.shuffle_fn(idxs)
        return idxs

    @delegates(plt.subplots)
    def show_batch(self, b=None, ctxs=None, max_n=9, nrows=3, ncols=3, figsize=(16, 10), **kwargs):
        b = self.one_batch()
        db = self.decode_batch(b, max_n=max_n)
        if figsize is None: figsize = (ncols*6, max_n//ncols*4)
        if ctxs is None: ctxs = get_grid(min(len(db), nrows*ncols), nrows=None, ncols=ncols, figsize=figsize, **kwargs)
        for i,ctx in enumerate(ctxs): show_tuple(db[i], ctx=ctx)

    @delegates(plt.subplots)
    def show_results(self, b, preds, ctxs=None, max_n=9, nrows=3, ncols=3, figsize=(16, 10), **kwargs):
        t = self.decode_batch(b, max_n=max_n)
        p = self.decode_batch((b[0],preds), max_n=max_n)
        if figsize is None: figsize = (ncols*6, max_n//ncols*4)
        if ctxs is None: ctxs = get_grid(min(len(t), nrows*ncols), nrows=None, ncols=ncols, figsize=figsize, **kwargs)
        for i,ctx in enumerate(ctxs):
            title = f'True: {t[i][1]}\nPred: {p[i][1]}'
            color = 'green' if t[i][1] == p[i][1] else 'red'
            t[i][0].show(ctx=ctx, title=title, title_color=color)

@delegates(plt.subplots)
def show_tuple(tup, **kwargs):
    "Display a timeseries plot from a decoded tuple"
    tup[0].show(title='unlabeled' if len(tup) == 1 else tup[1], **kwargs)

class TSDataLoader(NumpyDataLoader):
    @property
    def vars(self): return self.dataset[0][0].shape[-2]
    @property
    def len(self): return self.dataset[0][0].shape[-1]

# Cell
#tsai.data.core

_batch_tfms = ('after_item','before_batch','after_batch')

class NumpyDataLoaders(DataLoaders):
    _xblock = NumpyTensorBlock
    _dl_type = NumpyDataLoader
    def __init__(self, *loaders, path='.', device=default_device()):
        self.loaders,self.path = list(loaders),Path(path)
        self.device = device

    @classmethod
    @delegates(DataLoaders.from_dblock)
    def from_numpy(cls, X, y=None, splitter=None, valid_pct=0.2, seed=0, item_tfms=None, batch_tfms=None, **kwargs):
        "Create timeseries dataloaders from arrays (X and y, unless unlabeled)"
        if splitter is None: splitter = RandomSplitter(valid_pct=valid_pct, seed=seed)
        getters = [ItemGetter(0), ItemGetter(1)] if y is not None else [ItemGetter(0)]
        dblock = DataBlock(blocks=(cls._xblock, CategoryBlock),
                           getters=getters,
                           splitter=splitter,
                           item_tfms=item_tfms,
                           batch_tfms=batch_tfms)

        source = itemify(X) if y is None else itemify(X,y)
        return cls.from_dblock(dblock, source, **kwargs)

    @classmethod
    def from_dsets(cls, *ds, path='.', bs=64, num_workers=0, batch_tfms=None, device=None,
                   shuffle_train=True, **kwargs):
        default = (shuffle_train,) + (False,) * (len(ds)-1)
        defaults = {'shuffle': default, 'drop_last': default}
        kwargs = merge(defaults, {k: tuplify(v, match=ds) for k,v in kwargs.items()})
        kwargs = [{k: v[i] for k,v in kwargs.items()} for i in range_of(ds)]
        if not is_listy(bs): bs = [bs]
        if len(bs) != len(ds): bs = bs * len(ds)
        device = ifnone(device,default_device())
        return cls(*[cls._dl_type(d, bs=b, num_workers=num_workers, batch_tfms=batch_tfms, **k) \
                     for d,k,b in zip(ds, kwargs, bs)], path=path, device=device)

class TSDataLoaders(NumpyDataLoaders):
    _xblock = TSTensorBlock
    _dl_type = TSDataLoader

# Cell
#tsai.data.transform, slightly modified for optional discrete channels
class TSStandardize(Transform):
    "Standardize/destd batch of `NumpyTensor` or `TSTensor`"
    parameters, order = L('mean', 'std'), 99
    def __init__(self, mean=None, std=None, by_sample=False, by_var=False, verbose=False, discrete=False):
        self.mean = tensor(mean) if mean is not None else None
        self.std = tensor(std) if std is not None else None
        self.by_sample, self.by_var = by_sample, by_var
        if by_sample and by_var: self.axes = (2)
        elif by_sample: self.axes = (1, 2)
        elif by_var: self.axes = (0, 2)
        else: self.axes = ()
        self.verbose = verbose
        self.discrete=discrete

    @classmethod
    def from_stats(cls, mean, std): return cls(mean, std)

    def setups(self, dl: DataLoader):
        if self.mean is None or self.std is None:
            pv(f'{self.__class__.__name__} setup mean={self.mean}, std={self.std}, by_sample={self.by_sample}, by_var={self.by_var}', self.verbose)
#             x, *_ = dl.one_batch() ##??
#             assert not self.discrete or len(dl.ptls)==3
            x = dl.ptls[0] if not self.discrete else dl.ptls[1]## modification
            self.mean, self.std = x.float().mean(self.axes, keepdim=self.axes!=()), x.float().std(self.axes, keepdim=self.axes!=()) + 1e-7
            pv(f'mean: {self.mean}  std: {self.std}\n', self.verbose)

    def encodes(self, x:(NumpyTensor, TSTensor)):
        if self.discrete: return x
        pv('standardize cont encodes', self.verbose)
        if self.by_sample: self.mean, self.std = x.mean(self.axes, keepdim=self.axes!=()), x.std(self.axes, keepdim=self.axes!=()) + 1e-7
        return (x - self.mean) / self.std

    def encodes(self, x:(TSIntTensor)):
        if not self.discrete: return x
        pv('standardize int encodes', self.verbose)
        if self.by_sample: self.mean, self.std = x.mean(self.axes, keepdim=self.axes!=()), x.std(self.axes, keepdim=self.axes!=()) + 1e-7
        return (x - self.mean) / self.std

# Cell
@patch
def mul_min(x:(torch.Tensor, TSTensor, NumpyTensor), axes=(), keepdim=False):
    if axes == (): return retain_type(x.min(), x)
    axes = reversed(sorted(axes if is_listy(axes) else [axes]))
    min_x = x
    for ax in axes: min_x, _ = min_x.min(ax, keepdim)
    return retain_type(min_x, x)

@patch
def mul_max(x:(torch.Tensor, TSTensor, NumpyTensor), axes=(), keepdim=False):
    if axes == (): return retain_type(x.max(), x)
    axes = reversed(sorted(axes if is_listy(axes) else [axes]))
    max_x = x
    for ax in axes: max_x, _ = max_x.max(ax, keepdim)
    return retain_type(max_x, x)


class TSNormalize(Transform):
    "Normalize/denorm batch of `NumpyTensor` or `TSTensor`"
    parameters, order = L('min', 'max'), 99

    def __init__(self, min=None, max=None, range_min=-1, range_max=1, by_sample=True, by_var=False, verbose=False):
        self.min = tensor(min) if min is not None else None
        self.max = tensor(max) if max is not None else None
        self.range_min, self.range_max = range_min, range_max
        self.by_sample, self.by_var = by_sample, by_var
        if by_sample and by_var: self.axes = (2)
        elif by_sample: self.axes = (1, 2)
        elif by_var: self.axes = (0, 2)
        else: self.axes = ()
        self.verbose = verbose

    @classmethod
    def from_stats(cls, min, max, range_min=0, range_max=1): return cls(min, max, self.range_min, self.range_max)

    def setups(self, dl: DataLoader):
        if self.min is None or self.max is None:
            pv(f'{self.__class__.__name__} setup min={self.min}, max={self.max}, range_min={self.range_min}, range_max={self.range_max}, by_sample={self.by_sample}, by_var={self.by_var}',  self.verbose)
#             x, *_ = dl.one_batch()
            x = dl.ptls[0]
            self.min, self.max = x.mul_min(self.axes, keepdim=self.axes!=()), x.mul_max(self.axes, keepdim=self.axes!=())
            pv(f'min: {self.min}  max: {self.max}\n', self.verbose)

    def encodes(self, x:(NumpyTensor, TSTensor)):
        if self.by_sample: self.min, self.max = x.mul_min(self.axes, keepdim=self.axes!=()), x.mul_max(self.axes, keepdim=self.axes!=())
        return ((x - self.min) / (self.max - self.min)) * (self.range_max - self.range_min) + self.range_min

# Cell
def items_to_arrays(items):
    '''convert list of item tuples into X,y numpy arrays (for use with numpy dataloader)'''
#     return np.stack([x[0] for x in items]), np.stack([x[1] for x in items])
    return tuple(np.stack([x[i] for x in items]) for i in range(len(items[0])))