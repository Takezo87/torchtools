# AUTOGENERATED! DO NOT EDIT! File to edit: 01_models.ipynb (unless otherwise specified).

__all__ = ['noop', 'shortcut', 'Inception', 'InceptionBlock', 'InceptionTime', 'Squeeze', 'Unsqueeze', 'Add', 'Concat',
           'Permute', 'Transpose', 'View', 'Reshape', 'Max', 'LastStep', 'Noop', 'TransformerModel',
           'ScaledDotProductAttention', 'MultiHeadAttention', 'TSTEncoderLayer', 'TSTEncoder', 'TST', 'Sigmoid',
           'InceptionTimeSgmOld', 'InceptionTimeSgm', 'TransformerSgm', 'TransformerSgmD', 'InceptionTimeD',
           'InceptionTime_NH', 'InceptionTimeD_Mixed', 'InceptionTime_Mixed', 'TabNetTT', 'InceptionTimeVar',
           'nll_regression', 'nll_leaky_loss', 'qd_loss', 'InceptionTimeBounds']

# Cell
from .core import *

# Cell
import torch.nn as nn
import torch as torch
import torch.nn.functional as F

from functools import partial

from fastai.layers import SigmoidRange
from fastai.torch_basics import *
# from ..imports import *
# from .layers import *
# from .utils import *
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer

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
class Squeeze(Module):
    def __init__(self, dim=-1): self.dim = dim
    def forward(self, x): return x.squeeze(dim=self.dim)
    def __repr__(self): return f'{self.__class__.__name__}(dim={self.dim})'


class Unsqueeze(Module):
    def __init__(self, dim=-1): self.dim = dim
    def forward(self, x): return x.unsqueeze(dim=self.dim)
    def __repr__(self): return f'{self.__class__.__name__}(dim={self.dim})'


class Add(Module):
    def forward(self, x, y): return x.add(y)
    def __repr__(self): return f'{self.__class__.__name__}'


class Concat(Module):
    def __init__(self, dim=1): self.dim = dim
    def forward(self, *x): return torch.cat(*x, dim=self.dim)
    def __repr__(self): return f'{self.__class__.__name__}(dim={self.dim})'


class Permute(Module):
    def __init__(self, *dims): self.dims = dims
    def forward(self, x): return x.permute(self.dims)
    def __repr__(self): return f"{self.__class__.__name__}(dims={', '.join([str(d) for d in self.dims])})"


class Transpose(Module):
    def __init__(self, *dims, contiguous=False): self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)
    def __repr__(self):
        if self.contiguous: return f"{self.__class__.__name__}(dims={', '.join([str(d) for d in self.dims])}).contiguous()"
        else: return f"{self.__class__.__name__}({', '.join([str(d) for d in self.dims])})"


class View(Module):
    def __init__(self, *shape): self.shape = shape
    def forward(self, x): return x.view(x.shape[0], *self.shape)
    def __repr__(self): return f"{self.__class__.__name__}({', '.join(['bs'] + [str(s) for s in self.shape])})"


class Reshape(Module):
    def __init__(self, *shape): self.shape = shape
    def forward(self, x): return x.reshape(x.shape[0], *self.shape)
    def __repr__(self): return f"{self.__class__.__name__}({', '.join(['bs'] + [str(s) for s in self.shape])})"


class Max(Module):
    def __init__(self, dim=None, keepdim=False): self.dim, self.keepdim = dim, keepdim
    def forward(self, x): return x.max(self.dim, keepdim=self.keepdim)[0]
    def __repr__(self): return f'{self.__class__.__name__}(dim={self.dim}, keepdim={self.keepdim})'


class LastStep(Module):
    def forward(self, x): return x[..., -1]
    def __repr__(self): return f'{self.__class__.__name__}()'


Noop = nn.Sequential()

# Cell
class TransformerModel(Module):
    def __init__(self, c_in, c_out, d_model=64, n_head=1, d_ffn=128, dropout=0.1, activation="relu", n_layers=1):
        """
        Args:
            c_in: the number of features (aka variables, dimensions, channels) in the time series dataset
            c_out: the number of target classes
            d_model: total dimension of the model.
            nhead:  parallel attention heads.
            d_ffn: the dimension of the feedforward network model.
            dropout: a Dropout layer on attn_output_weights.
            activation: the activation function of intermediate layer, relu or gelu.
            num_layers: the number of sub-encoder-layers in the encoder.
        Input shape:
            bs (batch size) x nvars (aka variables, dimensions, channels) x seq_len (aka time steps)
            """
        self.permute = Permute(2, 0, 1)
        self.inlinear = nn.Linear(c_in, d_model)
        self.relu = nn.ReLU()
        encoder_layer = TransformerEncoderLayer(d_model, n_head, dim_feedforward=d_ffn, dropout=dropout, activation=activation)
        encoder_norm = nn.LayerNorm(d_model)
        self.transformer_encoder = TransformerEncoder(encoder_layer, n_layers, norm=encoder_norm)
        self.transpose = Transpose(1, 0)
        self.max = Max(1)
        self.outlinear = nn.Linear(d_model, c_out)

    def forward(self,x):
        x = self.permute(x) # bs x nvars x seq_len -> seq_len x bs x nvars
        x = self.inlinear(x) # seq_len x bs x nvars -> seq_len x bs x d_model
        x = self.relu(x)
        x = self.transformer_encoder(x)
        x = self.transpose(x) # seq_len x bs x d_model -> bs x seq_len x d_model
        x = self.max(x)
        x = self.relu(x)
        x = self.outlinear(x)
        return x

# Cell
class ScaledDotProductAttention(Module):
    def __init__(self, d_k:int): self.d_k = d_k
    def forward(self, q:Tensor, k:Tensor, v:Tensor, mask:Optional[Tensor]=None):

        # MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        scores = torch.matmul(q, k)                                         # scores : [bs x n_heads x q_len x q_len]

        # Scale
        scores = scores / (self.d_k ** 0.5)

        # Mask (optional)
        if mask is not None: scores.masked_fill_(mask, -1e9)

        # SoftMax
        attn = F.softmax(scores, dim=-1)                                    # attn   : [bs x n_heads x q_len x q_len]

        # MatMul (attn, v)
        context = torch.matmul(attn, v)                                     # context: [bs x n_heads x q_len x d_v]

        return context, attn

# Cell
class MultiHeadAttention(Module):
    def __init__(self, d_model:int, n_heads:int, d_k:int, d_v:int):
        r"""
        Input shape:  Q, K, V:[batch_size (bs) x q_len x d_model], mask:[q_len x q_len]
        """
        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)

        self.W_O = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, Q:Tensor, K:Tensor, V:Tensor, mask:Optional[Tensor]=None):

        bs = Q.size(0)

        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)       # q_s    : [bs x n_heads x q_len x d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)     # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)       # v_s    : [bs x n_heads x q_len x d_v]

        # Scaled Dot-Product Attention (multiple heads)
        context, attn = ScaledDotProductAttention(self.d_k)(q_s, k_s, v_s)          # context: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len]

        # Concat
        context = context.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v) # context: [bs x q_len x n_heads * d_v]

        # Linear
        output = self.W_O(context)                                                  # context: [bs x q_len x d_model]

        return output, attn

# Cell
class TSTEncoderLayer(Module):
    def __init__(self, d_model:int, n_heads:int, d_k:Optional[int]=None, d_v:Optional[int]=None, d_ff:int=256, res_dropout:float=0.1, activation:str="gelu"):

        assert d_model // n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = ifnone(d_k, d_model // n_heads)
        d_v = ifnone(d_v, d_model // n_heads)

        # Multi-Head attention
        self.self_attn = MultiHeadAttention(d_model, n_heads, d_k, d_v)

        # Add & Norm
        self.dropout_attn = nn.Dropout(res_dropout)
        self.batchnorm_attn = nn.BatchNorm1d(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), self._get_activation_fn(activation), nn.Linear(d_ff, d_model))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(res_dropout)
        self.batchnorm_ffn = nn.BatchNorm1d(d_model)

    def forward(self, src:Tensor, mask:Optional[Tensor]=None) -> Tensor:

        # Multi-Head attention sublayer
        ## Multi-Head attention
        src2, attn = self.self_attn(src, src, src, mask=mask)
        ## Add & Norm
        src = src + self.dropout_attn(src2) # Add: residual connection with residual dropout
        src = self.batchnorm_attn(src.permute(1,2,0)).permute(2,0,1) # Norm: batchnorm (requires d_model features to be in dim 1)

        # Feed-forward sublayer
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm
        src = src + self.dropout_ffn(src2) # Add: residual connection with residual dropout
        src = self.batchnorm_ffn(src.permute(1,2,0)).permute(2,0,1) # Norm: batchnorm (requires d_model features to be in dim 1)

        return src

    def _get_activation_fn(self, activation):
        if activation == "relu": return nn.ReLU()
        elif activation == "gelu": return nn.GELU()
        raise ValueError(f'{activation} is not available. You can use "relu" or "gelu"')

# Cell
class TSTEncoder(Module):
    def __init__(self, encoder_layer, n_layers):
        self.layers = nn.ModuleList([deepcopy(encoder_layer) for i in range(n_layers)])

    def forward(self, src:Tensor, mask:Optional[Tensor]=None) -> Tensor:
        output = src
        for mod in self.layers: output = mod(output, mask=mask)
        return output


# Cell
class TST(Module):
    def __init__(self, c_in:int, c_out:int, seq_len:int, max_seq_len:Optional[int]=None,
                 n_layers:int=3, d_model:int=128, n_heads:int=16, d_k:Optional[int]=None, d_v:Optional[int]=None,
                 d_ff:int=256, res_dropout:float=0.1, activation:str="gelu", fc_dropout:float=0.,
                 y_range:Optional[tuple]=None, verbose:bool=False, **kwargs):
        r"""TST (Time Series Transformer) is a Transformer that takes continuous time series as inputs.
        As mentioned in the paper, the input must be standardized by_var based on the entire training set.
        Args:
            c_in: the number of features (aka variables, dimensions, channels) in the time series dataset.
            c_out: the number of target classes.
            seq_len: number of time steps in the time series.
            max_seq_len: useful to control the temporal resolution in long time series to avoid memory issues.
            d_model: total dimension of the model (number of features created by the model)
            n_heads:  parallel attention heads.
            d_k: size of the learned linear projection of queries and keys in the MHA. Usual values: 16-512. Default: None -> (d_model/n_heads) = 32.
            d_v: size of the learned linear projection of values in the MHA. Usual values: 16-512. Default: None -> (d_model/n_heads) = 32.
            d_ff: the dimension of the feedforward network model.
            res_dropout: amount of residual dropout applied in the encoder.
            activation: the activation function of intermediate layer, relu or gelu.
            num_layers: the number of sub-encoder-layers in the encoder.
            fc_dropout: dropout applied to the final fully connected layer.
            y_range: range of possible y values (used in regression tasks).
            kwargs: nn.Conv1d kwargs. If not {}, a nn.Conv1d with those kwargs will be applied to original time series.
        Input shape:
            bs (batch size) x nvars (aka features, variables, dimensions, channels) x seq_len (aka time steps)
        """
        self.c_out, self.seq_len = c_out, seq_len

        # Input encoding
        q_len = seq_len
        self.new_q_len = False
        if max_seq_len is not None and seq_len > max_seq_len: # Control temporal resolution
            self.new_q_len = True
            q_len = max_seq_len
            tr_factor = math.ceil(seq_len / q_len)
            total_padding = (tr_factor * q_len - seq_len)
            padding = (total_padding // 2, total_padding - total_padding // 2)
            self.W_P = nn.Sequential(Pad1d(padding), Conv1d(c_in, d_model, kernel_size=tr_factor, stride=tr_factor))
            pv(f'temporal resolution modified: {seq_len} --> {q_len} time steps: kernel_size={tr_factor}, stride={tr_factor}, padding={padding}.\n', verbose)
        elif kwargs:
            self.new_q_len = True
            t = torch.rand(1, 1, seq_len)
            q_len = nn.Conv1d(1, 1, **kwargs)(t).shape[-1]
            self.W_P = nn.Conv1d(c_in, d_model, **kwargs) # Eq 2
            pv(f'Conv1d with kwargs={kwargs} applied to input to create input encodings\n', verbose)
        else:
            self.W_P = nn.Linear(c_in, d_model) # Eq 1: projection of feature vectors onto a d-dim vector space

        # Positional encoding
        W_pos = torch.normal(0, .1, (q_len, d_model), device=default_device())
        self.W_pos = nn.Parameter(W_pos, requires_grad=True)

        # Residual dropout
        self.res_dropout = nn.Dropout(res_dropout)

        # Encoder
        encoder_layer = TSTEncoderLayer(d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, res_dropout=res_dropout, activation=activation)
        self.encoder = TSTEncoder(encoder_layer, n_layers)
        self.flatten = Flatten()

        # Head
        self.head_nf = q_len * d_model
        self.head = self.create_head(self.head_nf, c_out, fc_dropout=fc_dropout, y_range=y_range)

    def create_head(self, nf, c_out, fc_dropout=0., y_range=None, **kwargs):
        layers = [nn.Dropout(fc_dropout)] if fc_dropout else []
        layers += [nn.Linear(nf, c_out)]
        if y_range: layers += [SigmoidRange(*y_range)]
        return nn.Sequential(*layers)


    def forward(self, x:Tensor, mask:Optional[Tensor]=None) -> Tensor:  # x: [bs x nvars x q_len]

        # Input encoding
        if self.new_q_len: u = self.W_P(x).transpose(2,1) # Eq 2        # u: [bs x d_model x q_len] transposed to [bs x q_len x d_model]
        else: u = self.W_P(x.transpose(2,1)) # Eq 1                     # u: [bs x q_len x d_model] transposed to [bs x q_len x d_model]

        # Positional encoding
        u = self.res_dropout(u + self.W_pos)

        # Encoder
        z = self.encoder(u)                                             # z: [bs x q_len x d_model]
        if self.flatten is not None: z = self.flatten(z)                # z: [bs x q_len * d_model]
        else: z = z.transpose(2,1).contiguous()                         # z: [bs x d_model x q_len]

        # Classification/ Regression head
        return self.head(z)                                             # output: [bs x c_out]

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
class TransformerSgm(nn.Module):
    '''
    add a sigmoid layer to Transformer to get the ouput in a certain range
    '''

    def __init__(self, n_in, n_out, seq_len=10, range=(-1,1), **kwargs):
        super().__init__()
        self.mod = nn.Sequential(TST(n_in, n_out, seq_len, **kwargs), SigmoidRange(*range))

    def forward(self, x):
        x = x.float()
        return self.mod(x)


# Cell
class TransformerSgmD(nn.Module):
    '''
    add a sigmoid layer to Transformer to get the ouput in a certain range
    discrete input channels
    '''

    def __init__(self, n_in, n_out, seq_len=10, range=(-1,1), **kwargs):
        super().__init__()
        self.mod = nn.Sequential(TST(n_in, n_out, seq_len, **kwargs), SigmoidRange(*range))

    def forward(self, xc, xd):
        xc, xd = TensorBase(xc), TensorBase(xd)
        x = torch.cat([xc.float(), xd.float()], dim=-2)
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
        #cast to TensorBase for pytorch 1.7 compatibility
        xc, xd = TensorBase(xc), TensorBase(xd)
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