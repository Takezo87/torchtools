# AUTOGENERATED! DO NOT EDIT! File to edit: 01_models.ipynb (unless otherwise specified).

__all__ = ['noop', 'shortcut', 'Inception', 'InceptionBlock', 'InceptionTime', 'Sigmoid', 'InceptionTimeSgmOld',
           'InceptionTimeSgm', 'InceptionTimeD', 'InceptionTime_NH', 'InceptionTimeD_Mixed', 'InceptionTime_Mixed',
           'TabNetTT', 'InceptionTimeVar', 'nll_regression', 'nll_leaky_loss', 'qd_loss', 'InceptionTimeBounds']

# Cell
from .core import *

# Cell
import torch.nn as nn
import torch as torch
import torch.nn.functional as F

from functools import partial

from fastai.layers import SigmoidRange

# Cell
# This is an unofficial PyTorch implementation by Ignacio Oguiza - oguiza@gmail.com based on:

# Fawaz, H. I., Lucas, B., Forestier, G., Pelletier, C., Schmidt, D. F., Weber, J., ... & Petitjean, F. (2019). InceptionTime: Finding AlexNet for Time Series Classification. arXiv preprint arXiv:1909.04939.
# Official InceptionTime tensorflow implementation: https://github.com/hfawaz/InceptionTime


def noop(x):
    return x

def shortcut(c_in, c_out):
    return nn.Sequential(*[nn.Conv1d(c_in, c_out, kernel_size=1),
                           nn.BatchNorm1d(c_out)])

class Inception(nn.Module):
    def __init__(self, c_in, bottleneck=32, ks=40, nb_filters=32):

        super().__init__()
        self.bottleneck = nn.Conv1d(c_in, bottleneck, 1) if bottleneck and c_in > 1 else noop
        mts_feat = bottleneck or c_in
        conv_layers = []
        kss = [ks // (2**i) for i in range(3)]
        # ensure odd kss until nn.Conv1d with padding='same' is available in pytorch 1.3
        kss = [ksi if ksi % 2 != 0 else ksi - 1 for ksi in kss]
        for i in range(len(kss)):
            conv_layers.append(
                nn.Conv1d(mts_feat, nb_filters, kernel_size=kss[i], padding=kss[i] // 2))
        self.conv_layers = nn.ModuleList(conv_layers)
        self.maxpool = nn.MaxPool1d(3, stride=1, padding=1)
        self.conv = nn.Conv1d(c_in, nb_filters, kernel_size=1)
        self.bn = nn.BatchNorm1d(nb_filters * 4)
        self.act = nn.ReLU()

    def forward(self, x):
        input_tensor = x.to(torch.float)
        x = self.bottleneck(input_tensor)
        for i in range(3):
            out_ = self.conv_layers[i](x)
            if i == 0: out = out_
            else: out = torch.cat((out, out_), 1)
        mp = self.conv(self.maxpool(input_tensor))
        inc_out = torch.cat((out, mp), 1)
        return self.act(self.bn(inc_out))


class InceptionBlock(nn.Module):
    def __init__(self,c_in,bottleneck=32,ks=40,nb_filters=32,residual=True,depth=6):

        super().__init__()

        self.residual = residual
        self.depth = depth

        #inception & residual layers
        inc_mods = []
        res_layers = []
        res = 0
        for d in range(depth):
            inc_mods.append(
                Inception(c_in if d == 0 else nb_filters * 4, bottleneck=bottleneck if d > 0 else 0,ks=ks,
                          nb_filters=nb_filters))
            if self.residual and d % 3 == 2:
                res_layers.append(shortcut(c_in if res == 0 else nb_filters * 4, nb_filters * 4))
                res += 1
            else: res_layer = res_layers.append(None)
        self.inc_mods = nn.ModuleList(inc_mods)
        self.res_layers = nn.ModuleList(res_layers)
        self.act = nn.ReLU()

    def forward(self, x):
        res = x
        for d, l in enumerate(range(self.depth)):
            x = self.inc_mods[d](x)
            if self.residual and d % 3 == 2:
                res = self.res_layers[d](res)
                x += res
                res = x
                x = self.act(x)
        return x

# Cell
class InceptionTime(nn.Module):
    def __init__(self,c_in,c_out,bottleneck=32,ks=40,nb_filters=32,residual=True,depth=6):
        super().__init__()
        self.block = InceptionBlock(c_in,bottleneck=bottleneck,ks=ks,nb_filters=nb_filters,
                                    residual=residual,depth=depth)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(nb_filters * 4, c_out)

    def forward(self, *x):
        x = torch.cat(x, dim=-2)
        x = self.block(x)
        x = self.gap(x).squeeze(-1)
        x = self.fc(x)
        return x

# Cell
class Sigmoid(nn.Module):
    '''
    sigmoid layer
    '''
    def __init__(self, low, high):
        super().__init__()
        self.high, self.low = high, low

    def forward(self, x):
        return torch.sigmoid(x)*(self.high-self.low)+self.low

# Cell
class InceptionTimeSgmOld(nn.Module):
    '''
    add a sigmoid layer to InceptionTime to get the ouput in a certain range
    '''

    def __init__(self, n_in, n_out):
        super().__init__()
        nn.Sequential()
        self.inc = InceptionTime(n_in, n_out)
        self.low, self.high = -1., 1.

    def forward(self, x):
        return torch.sigmoid(self.inc(x)) * (self.high - self.low) + self.low


# Cell
class InceptionTimeSgm(nn.Module):
    '''
    add a sigmoid layer to InceptionTime to get the ouput in a certain range
    '''

    def __init__(self, n_in, n_out, range=(-1,1)):
        super().__init__()
        self.mod = nn.Sequential(InceptionTime(n_in, n_out), SigmoidRange(*range))

    def forward(self, x):
        x = x.float()
        return self.mod(x)


# Cell
class InceptionTimeD(nn.Module):
    '''
    add a sigmoid layer to InceptionTime to get the ouput in a certain range
    '''

    def __init__(self, n_in, n_out):
        super().__init__()
        self.mod = nn.Sequential(InceptionTime(n_in, n_out), Sigmoid(-1., 1.))

    def forward(self, xc, xd):
        x = torch.cat([xc.float(), xd.float()], dim=-2)
        x = x.float()
#         print(f'InceptionTimeSgm dtype {x.dtype}')
        return self.mod(x)

# Cell
class InceptionTime_NH(nn.Module):
    '''inceptiontime, no final layer'''
    def __init__(self,c_in,c_out,bottleneck=32,ks=40,nb_filters=32,residual=True,depth=6):
        super().__init__()
        self.block = InceptionBlock(c_in,bottleneck=bottleneck,ks=ks,nb_filters=nb_filters,
                                    residual=residual,depth=depth)
        self.gap = nn.AdaptiveAvgPool1d(1)
#         self.fc = nn.Linear(nb_filters * 4, c_out)

    def forward(self, x):
        x = self.block(x)
#         print(x.shape)
        x = self.gap(x).squeeze(-1)
#         x = self.fc(x)
        return x

# Cell
def _map_xs(xs, xs_mask):
    '''
    xs: i-tuple of tensors
    xs_mask: length j>=i mask
    xs_id: lenght j>=i string list of x identifiers
    '''
    assert np.array(xs_mask).sum()==len(xs)
    res = np.array([None]*len(xs_mask))
    res[np.where(xs_mask)[0]]=xs
    return res

# Cell
class InceptionTimeD_Mixed(nn.Module):
    '''
    mixed model for timeseries and tabular data
    ts_mod: InceptionTime model without final fully connected lay
    tab_mod: MLP or TabNet, currently both cont and cat is required
    outputs are concatenated, then put through a fully connected layer, then sigmoid range
    '''

    def __init__(self, n_c, n_d, n_out, n_cont, emb_szs=None):
        super().__init__()
        self.n_c, self.n_d, self.n_cont, self.emb_szs = n_c, n_d, n_out, emb_szs
        assert n_c>0, 'at least one continuous channel required'
        self.ts_mod = InceptionTime_NH(n_c+n_d, n_out) #128
        self.sgm = Sigmoid(-1,1)
#         self.mod = nn.Sequential(InceptionTime(n_in, n_out), Sigmoid(-1., 1.))
#         self.tab_mod = nn.Sequential(nn.Linear(2,100), nn.ReLU(), nn.Linear(100,64))
        self.tab_mod = TabNetModel(emb_szs=emb_szs, n_cont=n_cont, out_sz=64)
        self.fc = nn.Linear(192,n_out)

#     def forward(self, xc, xd, xt, xcat=None):
    def forward(self, *xs):


        xs_mask = [self.n_c>0, self.n_d>0, self.n_cont>0, len(self.emb_szs)>0]
#         x_type_idxs = [i for i in range(4) if has_x[i]]
        xc,xd,xt,xcat = map_xs(xs, xs_mask)

        x_ts=xc.float()
        if xd is not None: x_ts = torch.cat([x_ts, xd.float()], dim=-2)

#         x_ts=torch.cat([xs[0].float(), xd.float()], dim=-2) if self.n_d>0 else x_ts


#         x = t
#         x = x.float()
#         print(f'InceptionTimeSgm dtype {x.dtype}')
#         print(self.ts_mod(x).shape, self.tab_mod(xt.float().squeeze(-2)).shape )
        xcat=xcat.long() if xcat is not None else None
        xt=xt.float() if xt is not None else None
        x_all = torch.cat([self.ts_mod(x_ts), self.tab_mod(xcat, xt)], dim=-1)
        return self.sgm(self.fc(x_all))

# Cell
class InceptionTime_Mixed(nn.Module):
    '''
    mixed model for timeseries and tabular data
    ts_mod: InceptionTime model without final fully connected lay
    tab_mod: MLP or TabNet, currently both cont and cat is required
    outputs are concatenated, then put through a fully connected layer, no sigmoid for classification
    '''

    def __init__(self, n_c, n_d, n_out, n_cont, emb_szs=None):
        super().__init__()
        self.n_c, self.n_d, self.n_cont, self.emb_szs = n_c, n_d, n_out, emb_szs
        assert n_c>0, 'at least one continuous channel required'
        self.ts_mod = InceptionTime_NH(n_c+n_d, n_out) #128
#         self.mod = nn.Sequential(InceptionTime(n_in, n_out), Sigmoid(-1., 1.))
#         self.tab_mod = nn.Sequential(nn.Linear(2,100), nn.ReLU(), nn.Linear(100,64))
        self.tab_mod = TabNetModel(emb_szs=emb_szs, n_cont=n_cont, out_sz=64)
        self.fc = nn.Linear(192,n_out)

#     def forward(self, xc, xd, xt, xcat=None):
    def forward(self, *xs):


        xs_mask = [self.n_c>0, self.n_d>0, self.n_cont>0, len(self.emb_szs)>0]
#         x_type_idxs = [i for i in range(4) if has_x[i]]
        xc,xd,xt,xcat = map_xs(xs, xs_mask)

        x_ts=xc.float()
        if xd is not None: x_ts = torch.cat([x_ts, xd.float()], dim=-2)

#         x_ts=torch.cat([xs[0].float(), xd.float()], dim=-2) if self.n_d>0 else x_ts


#         x = t
#         x = x.float()
#         print(f'InceptionTimeSgm dtype {x.dtype}')
#         print(self.ts_mod(x).shape, self.tab_mod(xt.float().squeeze(-2)).shape )
        xcat=xcat.long() if xcat is not None else None
        xt=xt.float() if xt is not None else None
        x_all = torch.cat([self.ts_mod(x_ts), self.tab_mod(xcat, xt)], dim=-1)
        return self.fc(x_all)

# Cell
class TabNetTT(nn.Module):
    '''
    convenience wrapper for pure TabNetModel models
    '''
    def __init__(self, emb_szs, n_cont, out_sz, **kwargs):
        super().__init__()
        self.tab = TabNetModel(emb_szs, n_cont, out_sz, **kwargs)

    def forward(self, xt, xcat):
        xcat=xcat.long() if xcat is not None else None
        xt=xt.float() if xt is not None else None
        return self.tab(xcat, xt)

# Cell
class InceptionTimeVar(nn.Module):
    '''
    output mean and variance
    regression model, sigmoid for the mean output optional
    '''

    def __init__(self, n_in, n_out, meanrange=None):
        super().__init__()
        models  = [InceptionTime(n_in, n_out+1)]
        if meanrange:
            self.sigmoid = Sigmoid(*meanrange)
        self.mod = nn.Sequential(*models)

    def forward(self, x):
        x = x.float()
        output = self.mod(x)
        ## enforce positivity of sigma^2
        ##output_sig_pos = tf.log(1 + tf.exp(output_sig)) + 1e-06
#         output[:,-1] = (output[:,-1].exp()+1).log_() + 1e-06
        output[:,-1] = F.softplus(output[:,-1].clone())

        if getattr(self, 'sigmoid', None): output[:,:-1] = self.sigmoid(output[:,:-1])
        return output


# Cell
def nll_regression(preds, y_true, c=5):
    '''
    negative log likelihood loss for regression, both mu and sigma are predicted

    Simple and Scalable Predictive UncertaintyEstimation using Deep Ensembles
    Balaji Lakshminarayanan, Alexander Pritzel, Charles Blundell, DeepMind

    '''

    s1 = 0.5*preds[:,1].log()
    s2 = 0.5*(yb.squeeze()-preds[:,0]).pow(2).div(preds[:,1])
    loss = (s1+s2).mean() + c
    return loss

# Cell
def nll_leaky_loss(preds, y_true, c=5, alpha=0.5):
    '''
    leaky_loss with variance

    Simple and Scalable Predictive UncertaintyEstimation using Deep Ensembles
    Balaji Lakshminarayanan, Alexander Pritzel, Charles Blundell, DeepMind

    '''

    s1 = 0.5*preds[:,1].log()
    l1 = -F.leaky_relu(preds[:,0], alpha)*y_true.float().squeeze()
    s2 = 0.5*(l1.div(preds[:,1]+1)) ## +1 to prevent optimizing for variance, maybe just an artifical problem
    loss = (s1+s2).mean() + c
    return loss

# Cell
def qd_loss(preds, y_true, alpha=0.4, l=0.01, s=0.01, add=False, slope=1.):
    '''
    qd loss implementation adapted for "leaky loss problems"
    preds: predictions for both lower and upper bounds
    alpha: confidence intervall parameter, different from alpha in leaky_loss
    s: smoothing factor for sigmoid
    l: agrangian controlling width vs coverage (default in the paper impl. is 0.01 which seems lowI)
    '''
    ll = lambda x: F.leaky_relu(x, negative_slope=slope)

    y_lower = preds[:,0].clone()
    y_upper = preds[:,1].clone() if not add else y_lower+preds[:,1]

#     if not add:
#         y_lower, y_upper = preds[:, 0].clone(), preds[:, 1].clone()
#     else:
#         y_lower, y_upper = preds[:, 0].clone(), preds[:,0].clone()+preds[:, 1].clone()
# #     hard counts, how many of the predictions have the right sign?
    khu = (torch.sign(y_upper*y_true) > 0).int()
    khl = (torch.sign(y_lower*y_true) > 0).int()

#     return preds.mean()
    # soft counts, sign step function replaced by a smoother sigmoid

    ksu = torch.sigmoid((ll(y_upper)*y_true)*s)
    ksl = torch.sigmoid((y_true*ll(y_lower))*s)
    kh,ks = khu*khl, ksu*ksl
#     print(kh)
#     print(kh.sum(), ks.sum())

    #mpiw: mean predicted interval width
    f = 1/(kh.sum()+1e-6)
#     print((y_upper-y_lower))
    mpiw = ((y_upper-y_lower)*kh).sum()*f

    #picp: predicted interval coverage probability
    picp_s = ks.mean()

    print(f'mpiw {mpiw}, pcip_soft: {picp_s}')
    s2 = l*preds.shape[0]/(alpha*(1-alpha))
    s3 = torch.max(torch.zeros(1, device=preds.device), picp_s).pow(2)
    loss_s = mpiw + l*preds.shape[0]/(alpha*(1-alpha)) * torch.max(torch.zeros(1, device=preds.device),
                                                                   picp_s).pow(2)
    return loss_s

# Cell
class InceptionTimeBounds(nn.Module):
    '''
    use InceptionTimeVar implementation for bounds
    output[:, -1] is positive and y_upper corresponds to output[:,0]+output[:,1] --> loss
    '''

    def __init__(self, n_in, n_out, meanrange=None):
        super().__init__()
        models  = [InceptionTime(n_in, n_out+1)]
        if meanrange:
            self.sigmoid = Sigmoid(*meanrange)
        self.mod = nn.Sequential(*models)

    def forward(self, x):
        x = x.float()
        output = self.mod(x)
        ## enforce positivity of sigma^2
        ##output_sig_pos = tf.log(1 + tf.exp(output_sig)) + 1e-06
#         output[:,-1] = (output[:,-1].exp()+1).log_() + 1e-06
        output[:,-1] = F.softplus(output[:,-1].clone())  ## autograd problems when not using clone, why???

        if getattr(self, 'sigmoid', None): output[:,:-1] = self.sigmoid(output[:,:-1])
        return output