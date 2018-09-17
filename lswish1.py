import numpy

import chainer
from chainer.backends import cuda
from chainer import function_node
from chainer import utils
from chainer.utils import type_check

if cuda.cudnn_enabled:
    cudnn = cuda.cudnn
    libcudnn = cuda.cuda.cudnn
    _mode = libcudnn.CUDNN_ACTIVATION_SIGMOID


class LeakySwish1(function_node.FunctionNode):

    """Logistic sigmoid function."""

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward_cpu(self, inputs):
        
        x = inputs[0].copy()
        self.retain_inputs((0,))
        
        half = x.dtype.type(0.5)
        s = y.dtype.type(0.2)
        y = utils.force_array((1 - s) * (numpy.tanh(x * half) * half + half) + s)
        self.retain_outputs((0,))
        self._use_cudnn = False
        return y,

    def forward_gpu(self, inputs):
        #self.retain_inputs((0,))
        x = inputs[0].copy()
        self.retain_inputs((0,))
        if False:
        #if chainer.should_use_cudnn('==always') and x.flags.c_contiguous:
            y = cudnn.activation_forward(x, _mode)
            self.retain_inputs((0,))
            self._use_cudnn = True
        else:
            
            #self.retain_inputs((0,))
            y = cuda.elementwise(
                'T x', 'T y', 'y = x * ((1 - 0.2) * (tanh(x * 0.5) * 0.5 + 0.5) + 0.2)',
                'leaky_swish_fwd')(x)
            #self.retain_inputs((0,))
            self._use_cudnn = False

        self.retain_outputs((0,))
        return y,

    def backward(self, indexes, grad_outputs):
        #if self._use_cudnn:
        #    x = self.get_retained_inputs()[0].data
        #else:
        #    x = None
        x = self.get_retained_inputs()[0]
        y = self.get_retained_outputs()[0]
        gy, = grad_outputs
        #print(LeakySwish1Grad((x,)).apply((y, gy, x)))
        #return LeakySwish1Grad((x,)).apply((y, gy))
        return LeakySwish1Grad((x,)).apply((y, gy, x))


class LeakySwish1Grad(function_node.FunctionNode):

    """Logistic sigmoid gradient function."""

    def __init__(self, inputs):
        super(LeakySwish1Grad, self).__init__()
        self.x = inputs[0]

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 3)
        type_check.expect(in_types[0].dtype.kind == 'f')
        type_check.expect(in_types[1].dtype.kind == 'f')
        type_check.expect(in_types[2].dtype.kind == 'f')

    def forward_cpu(self, inputs):
        self.retain_inputs((0, 1, 2))
        y, gy, x = inputs
        #x = self.x
        one = y.dtype.type(1)
        s = y.dtype.type(0.2)
        #print(utils.force_array(gy * ((one - s) * (y + x * y * (one - y)) + s)))
        #return utils.force_array(gy * y * (one - y)),
        return utils.force_array(gy * ((one - s) * (y + x * y * (one - y)) + s)),

    def forward_gpu(self, inputs):
        self.retain_inputs((0, 1, 2))
        y, gy, x = inputs
        #x = self.x
        if False:
        #if (chainer.should_use_cudnn('==always') and gy.flags.c_contiguous and
                #self.x is not None and self.x.flags.c_contiguous):
            gx = cudnn.activation_backward(self.x, y, gy, _mode)
        else:
            gx = cuda.elementwise(
                'T x, T y, T gy', 'T gx',
                'gx = gy * ((1 - 0.2) * (y + x * y * (1 - y)) + 0.2)',
                #'gx = gy * y * (1 - y)',
                'leaky_swish_bwd')(x, y, gy)
        
        return gx,

    def backward(self, indexes, grad_outputs):
        y, gy, x = self.get_retained_inputs()
        #x = self.x
        ggx, = grad_outputs
        
        #return ggx * gy * (1 - 2 * y), ggx * y * (1 - y)
        return ggx * 0.8 * (y * (1 - y) * (2 + x * (1 - 2 * y))), ggx * y * (1 - y)


def leaky_swish_1(x):
    """Element-wise sigmoid logistic function.
     .. math:: f(x)=(1 + \\exp(-x))^{-1}.
    Args:
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`):
            Input variable. A :math:`(s_1, s_2, ..., s_N)`-shaped float array.
    Returns:
        ~chainer.Variable: Output variable. A
        :math:`(s_1, s_2, ..., s_N)`-shaped float array.
    .. admonition:: Example
        It maps the input values into the range of :math:`[0, 1]`.
        >>> x = np.arange(-2, 3, 2).astype(np.float32)
        >>> x
        array([-2.,  0.,  2.], dtype=float32)
        >>> F.sigmoid(x)
        variable([0.11920291, 0.5       , 0.8807971 ])
    """
    y, = LeakySwish1().apply((x,))
    return y
