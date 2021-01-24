# AUTOGENERATED! DO NOT EDIT! File to edit: 00_core.ipynb (unless otherwise specified).

__all__ = ['test', 'leaky_loss', 'gamblers_loss', 'leaky_loss_2d', 'one_hot', 'create_rww_categorical_crossentropy',
           'ahc_fp_weights', 'ahc_rww_loss', 'unweighted_profit', 'unweighted_profit_05', 'weighted_profit',
           'get_loss_fn', 'get_loss_fn_class', 'is_array', 'listify', 'map_xs', 'FixedSplitter', 'TSSplitter']

# Cell
#hide
import torch.nn.functional as F
import torch as torch
from functools import partial
from fastai.basics import *

# Cell
def test():
    '''a test function'''
    print('test')

# Cell
def leaky_loss(preds, y_true, alpha=0.05):
    '''
    objective function, including negative predictions with factor alpha
    '''
    loss_1 = (F.leaky_relu(preds, alpha).squeeze()*y_true.float()).mean()*(-1)
    #loss_1.requires_grad_()
    #assert loss_1.requires_grad == True
    # loss_1.requires_grad_(True)
    return loss_1

# Cell
def gamblers_loss(preds, y_true, o=2):
    '''
    regression adaption of gambler's loss
    o is a hyperparameter
    '''
    #
    preds = F.softmax(preds, dim=1)
    outputs, reservation = preds[:, :-1].squeeze(), preds[:, -1]
    # print(outputs, reservation)
    # gain = torch.gather(outputs, dim=1, index=targets.unsqueeze(1)).squeeze()
    doubling_rate = (outputs*y_true + reservation).log()
    # print(doubling_rate)
    return - doubling_rate.mean()

# Cell
def leaky_loss_2d(preds, y_true, alpha=0.05, weights=None):
    '''
    objective function, including negative predictions with factor alpha
    weights: target variable weights
    '''
    assert len(y_true.shape)==2, 'y_true needs to be 2d'
     # weight of the first y-value
    prod = (F.leaky_relu(preds, alpha).squeeze()*y_true.float())
    print(prod)
    if weights:
        prod.mul_(torch.tensor(weights)[:, None])
#     print(prod)
    loss_1 = prod.mean()*(-1)
    loss_1.requires_grad_(True)
    return loss_1


# Cell
def one_hot(t, k=5):
    '''
    one-hot enconcoding of t with k values
    '''
    ohc = torch.zeros(t.shape[0], k)
    ohc[range(t.shape[0]), t]=1
    return ohc.to(t.device)

# Cell
def create_rww_categorical_crossentropy(k, loss_type, fn_weights=None, fp_weights=None, return_weights=False):
    """Real-World-Weighted crossentropy between an output tensor and a target tensor.

    The loss_types other than rww_categorical_crossentropy reimplement existing
    functions in Keras but are not as well optimized.
    These loss_types are usable directly, but, are more useful when calling
    return_weights=True, which then returns fn and fp weights matrixes of size (k,k).
    Editing those to reflect real world costs, then passing them back into
    create_rww_crossentropy with loss_type "rww_crossentropy" is the recommended approach.

    Example Usage:

    Suppose you have three classes: cat, dog, and other.

    Cat is one-hot encoded as [1,0,0], dog as [0,1,0], other as [0,0,1]

    The the following code increases the incremental penalty of
    mislabeling a true target 0 (cat) with a false label 1 (dog) at a cost of 99,
    versus the default of zero. Note that the existing fn_weights also has a
    default cost of 1 for missing the true target of 1, for a total cost of
    100 versus the default cost of 1.

    fn_weights, fp_weights = create_rww_categorical_crossentropy(10, "categorical_crossentropy", return_weights=True)
    fp_weights[0, 1] = 99
    loss = create_rww_categorical_crossentropy(10, "rww_crossentropy", fn_weights, fp_weights)

    ...

    The fn and fp weights are easy to reason about.

    fn_weights is [x1, __, __]
                [__, x2, __]
                [__, __, x3]

    x1 represents the scale of the cost for a fn for cat, x2 for dog, and x3 for other.

    This is calculated as fn_weight * log(y_pred).

    In the case of loss_type=categorical_crossentropy,
    x1, x2, and x3 all equal the value one.
    All elements not on the main axis must equal zero.

    Note that fn_weights could have been represented as a vector,
    not a matrix, however, we use a matrix to keep symmetry with
    fp_weights, and, to prepare for
    multi-label classification.

    ...

    fp_weights is concerned with the costs of the fps from the other classes.

    fp_weights of [__, x1, x2]
                [x3, __, x4]
                [x5, x6, __]

    x1 represents the cost of predicting 1 for dog, when it should be 0 for cat.
    x2 represents predicting 2 for other, when the target is 0 for cat.
    x3 represents predicing 0 for cat, when the target is 1 for dog.
    etc.

    Args:
    * k: 2 or more for number of categories, including "other".
    * loss_type: "categorical_crossentropy" to initialize to
      standard softmax_crossentropy behavior,
      or "weighted_categorical_crossentropy" for standard behavior, or,
      or "rww_crossentropy" for full weight matrix of all possible fn/fp combinations.
    * fn_weights: a numpy array of shape (k,k). The main diagonal can
      contain non-zero values; all other values must be zero.
    * fp_weights: a numpy array of shape (k,k) to define specific combinations
      of false positive. The main diag should be zeros.
    * return_weights: If False (default), returns cost function. If True,
      returns fn and fp weights as np.array.
    Returns:
    * retval: Loss function for use Keras.model.fit, or if return_weights
      arg is True, the fn_weights and fp_weights matrixes.
    """

    full_fn_weights = None
    full_fp_weights = None

    anti_eye = np.ones((k,k)) - np.eye(k)

    if (loss_type=="categorical_crossentropy"):
        full_fn_weights = np.identity((k))
        full_fp_weights = np.zeros((k, k)) # Softmax crossentropy ignores fp.

    elif(loss_type=="weighted_categorical_crossentropy"):
        full_fn_weights = np.eye(k) * fn_weights
        full_fp_weights = np.zeros((k, k)) # softmax crossentropy ignores fp

    elif(loss_type=="rww_crossentropy"):
#         assert not np.count_nonzero(fn_weights * anti_eye)
#         assert not np.count_nonzero(fp_weights * np.eye(k))

        full_fn_weights = fn_weights
        # Novel piece: allow any combination of fp.
        full_fp_weights = fp_weights

    else:
        raise Exception("unknown loss_type: " + str(loss_type))


    fn_wt = tensor(full_fn_weights)
    fp_wt = tensor(full_fp_weights)
#     fn_wt = K.constant(full_fn_weights) # (k,k), always sparse along main diag.
#     fp_wt = K.constant(full_fp_weights) # (k,k), always dense except main diag.

    def loss_function(preds, y_true):
        '''
        '''
#     output = torch.clip(output, K.epsilon(), 1 - K.epsilon())
        output = F.log_softmax(preds, dim=-1)
        target = one_hot(y_true)


        logs = output
        logs_1_sub  = (1-F.softmax(preds, dim=-1)).log()    #     logs = K.log(output) # shape (m, k), dense. 1 is good.
    #     logs_1_sub = K.log(1-output) # shape (m, k), dense. 0 is good.
#         print(target.shape, fp_wt.shape)
        m_full_fn_weights = target.matmul(fn_wt) # (m,k) . (k, k)
        m_full_fp_weights = target.matmul(fp_wt) # (m,k) . (k, k)

        return - torch.mean(m_full_fn_weights * logs +
                        m_full_fp_weights * logs_1_sub)

    if (return_weights):
        return full_fn_weights, full_fp_weights
    else:
        return loss_function

# Cell
def ahc_fp_weights(w=10):
    '''
    rww weight matrix for false positives
    tensor([[ 0.,  0.,  1., 10.,  1.],
        [ 0.,  0.,  1., 10.,  1.],
        [ 1.,  1.,  0.,  1.,  1.],
        [10., 10.,  1.,  0.,  0.],
        [10., 10.,  1.,  0.,  0.]])
    '''
    fp_w = torch.ones(5,5)
    for i in range(5):
        fp_w[i,i] = 0
    fp_w[0,1]=0
    fp_w[0,3]=w
    fp_w[0,3]=w
    fp_w[1,0]=0
    fp_w[1,3]=w
    fp_w[1,3]=w
    fp_w[3,4]=0
    fp_w[3,0]=w
    fp_w[3,1]=w
    fp_w[4,3]=0
    fp_w[4,0]=w
    fp_w[4,1]=w
    return fp_w

# Cell
def ahc_rww_loss(weight=tensor([1.,10.,1.,10.,1.]), w=10):
#     fp_weights = ahc_fp_weights().to(default_device())
    fp_weights = ahc_fp_weights()
#     fn_weights = (weight*torch.eye(5)).to(default_device())
    fn_weights = (weight*torch.eye(5))
#     print(fn_weights)
    fp=fp_weights.to(default_device())
    fn=fn_weights.to(default_device())
    loss = create_rww_categorical_crossentropy(5, 'rww_crossentropy',
                                               return_weights=False, fn_weights=fn,
                                               fp_weights=fp)
    return loss

# Cell
def unweighted_profit(preds, y_true, threshold=0):
    '''
    metric, negative predictions ignored, y_true of positive predictions equally weighted
    '''
    m_value = ((preds.squeeze()>threshold).float()*y_true.float()).mean()
    return m_value

# Cell
def unweighted_profit_05(preds, y_true, threshold=0.5):
    '''
    metric, negative predictions ignored, y_true of positive predictions equally weighted
    '''
    m_value = ((preds.squeeze()>threshold).float()*y_true.float()).mean()
    return m_value

# Cell
def weighted_profit(preds, y_true, threshold=0):
    '''
    metric, negative predictions ignored, results weighted by positive predictions
    adding threshold possible
    '''
    loss_1 = ((preds.squeeze()>threshold).float()*(preds.squeeze())*y_true.float()).mean()
    return loss_1

# Cell
def get_loss_fn(loss_fn_name, **kwargs):
    '''
    wrapper to create a partial with a more convenient __name__ attribute
    '''
    if loss_fn_name == 'leaky_loss':
        assert kwargs.get('alpha', None) is not None, 'need to specify alpha with leaky_loss'
        _loss_fn = partial(leaky_loss, alpha=kwargs['alpha'])
        _loss_fn.__name__ = loss_fn_name
        return _loss_fn
    if loss_fn_name == 'gambler':
        return gamblers_loss
    return None

# Cell
def get_loss_fn_class(loss_fn_name, weight=None):
    '''
    loss function buildier for classification tasks
    '''
#     weights = tensor([1., 10., 1, .10, 1.])
    if loss_fn_name == 'rww': return ahc_rww_loss()
    else:
        print(f'crosse entropy weigts {weight}')
        return CrossEntropyLossFlat() if weight is None else CrossEntropyLossFlat(weight=weight.to(device))

# Cell
#fastcore.foundations
def is_array(x): return hasattr(x,'__array__') or hasattr(x,'iloc')
def listify(o):
    if o is None: return []
    if isinstance(o, list): return o
    if isinstance(o, str) or is_array(o): return [o]
    if is_iter(o): return list(o)
    return [o]

# Cell
def map_xs(xs, xs_mask):
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
def FixedSplitter(end_train=10000, end_valid=15000):
    def _inner(o, **kwargs):
        return L(range(0, end_train)), L(range(end_train, end_valid))
    return _inner

# Cell
def TSSplitter(train_perc=0.8, test=False):
    def _inner(o, **kwargs):
        l = len(o)
        end_train = int(l*train_perc)
        end_val = l if not test else int(l*(train_perc+(1-train_perc)*0.5))
        end_test = l
        if test: return L(range(end_train), range(end_train, end_val), range(end_val, end_test))
        return L(range(end_train), range(end_train, end_val))
    return _inner