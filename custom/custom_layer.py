from chainer.functions.connection import convolution_2d, linear
from chainer import initializers
from chainer import link
from chainer.utils import argument
from chainer import variable
from chainer.functions import broadcast_to, bias

import functools
import operator

import numpy

class EqualizedConvolution2d(link.Link):
    
    """__init__(self, in_channels, out_channels, ksize=None, stride=1, pad=0, nobias=False, initialW=None, initial_bias=None, *, dilate=1, groups=1)
    Two-dimensional convolutional layer.
    This link wraps the :func:`~chainer.functions.convolution_2d` function and
    holds the filter weight and bias vector as parameters.
    The output of this function can be non-deterministic when it uses cuDNN.
    If ``chainer.configuration.config.deterministic`` is ``True`` and
    cuDNN version is >= v3, it forces cuDNN to use a deterministic algorithm.
    Convolution links can use a feature of cuDNN called autotuning, which
    selects the most efficient CNN algorithm for images of fixed-size, 
    can provide a significant performance boost for fixed neural nets.
    To enable, set `chainer.using_config('autotune', True)`
    .. warning::
        ``deterministic`` argument is not supported anymore since v2.
        Instead, use ``chainer.using_config('cudnn_deterministic', value``
        (value is either ``True`` or ``False``).
        See :func:`chainer.using_config`.
    Args:
        in_channels (int or None): Number of channels of input arrays.
            If ``None``, parameter initialization will be deferred until the
            first forward data pass at which time the size will be determined.
        out_channels (int): Number of channels of output arrays.
        ksize (int or pair of ints): Size of filters (a.k.a. kernels).
            ``ksize=k`` and ``ksize=(k, k)`` are equivalent.
        stride (int or pair of ints): Stride of filter applications.
            ``stride=s`` and ``stride=(s, s)`` are equivalent.
        pad (int or pair of ints): Spatial padding width for input arrays.
            ``pad=p`` and ``pad=(p, p)`` are equivalent.
        nobias (bool): If ``True``, then this link does not use the bias term.
        initialW (:ref:`initializer <initializer>`): Initializer to
            initialize the weight. When it is :class:`numpy.ndarray`,
            its ``ndim`` should be 4.
        initial_bias (:ref:`initializer <initializer>`): Initializer to
            initialize the bias. If ``None``, the bias will be initialized to
            zero. When it is :class:`numpy.ndarray`, its ``ndim`` should be 1.
        dilate (int or pair of ints):
            Dilation factor of filter applications.
            ``dilate=d`` and ``dilate=(d, d)`` are equivalent.
        groups (:class:`int`): Number of groups of channels. If the number
            is greater than 1, input tensor :math:`W` is divided into some
            blocks by this value channel-wise. For each tensor blocks,
            convolution operation will be executed independently. Input channel
            size ``in_channels`` and output channel size ``out_channels`` must
            be exactly divisible by this value.
    .. seealso::
       See :func:`chainer.functions.convolution_2d` for the definition of
       two-dimensional convolution.
    Attributes:
        W (~chainer.Variable): Weight parameter.
        b (~chainer.Variable): Bias parameter.
    .. admonition:: Example
        There are several ways to make a Convolution2D link.
        Let an input vector ``x`` be:
        >>> x = np.arange(1 * 3 * 10 * 10, dtype=np.float32).reshape(1, 3, 10, 10)
        1. Give the first three arguments explicitly:
            >>> l = L.Convolution2D(3, 7, 5)
            >>> y = l(x)
            >>> y.shape
            (1, 7, 6, 6)
        2. Omit ``in_channels`` or fill it with ``None``:
            The below two cases are the same.
            >>> l = L.Convolution2D(7, 5)
            >>> y = l(x)
            >>> y.shape
            (1, 7, 6, 6)
            >>> l = L.Convolution2D(None, 7, 5)
            >>> y = l(x)
            >>> y.shape
            (1, 7, 6, 6)
            When you omit the first argument, you need to specify the other
            subsequent arguments from ``stride`` as keyword auguments. So the
            below two cases are the same.
            >>> l = L.Convolution2D(7, 5, stride=1, pad=0)
            >>> y = l(x)
            >>> y.shape
            (1, 7, 6, 6)
            >>> l = L.Convolution2D(None, 7, 5, 1, 0)
            >>> y = l(x)
            >>> y.shape
            (1, 7, 6, 6)
    """  # NOQA
    
    def __init__(self, in_channels, out_channels, ksize=None, stride=1, pad=0,
                 nobias=False, initialW=None, initial_bias=None, **kwargs):
        super(EqualizedConvolution2d, self).__init__()

        argument.check_unexpected_kwargs(
            kwargs, deterministic="deterministic argument is not "
            "supported anymore. "
            "Use chainer.using_config('cudnn_deterministic', value) "
            "context where value is either `True` or `False`.")
        dilate, groups = argument.parse_kwargs(kwargs,
                                               ('dilate', 1), ('groups', 1))

        if ksize is None:
            out_channels, ksize, in_channels = in_channels, out_channels, None

        self.ksize = ksize
        self.stride = _pair(stride)
        self.pad = _pair(pad)
        self.dilate = _pair(dilate)
        self.out_channels = out_channels
        self.groups = int(groups)

        with self.init_scope():
            W_initializer = initializers._get_initializer(initialW)
            self.W = variable.Parameter(W_initializer)
            
            if in_channels is not None:
                self._initialize_params(in_channels)

            if nobias:
                self.b = None
            else:
                if initial_bias is None:
                    initial_bias = 0
                bias_initializer = initializers._get_initializer(initial_bias)
                self.b = variable.Parameter(bias_initializer, out_channels)

    def _initialize_params(self, in_channels):
        kh, kw = _pair(self.ksize)
        if self.out_channels % self.groups != 0:
            raise ValueError('the number of output channels must be'
                             ' divisible by the number of groups')
        if in_channels % self.groups != 0:
            raise ValueError('the number of input channels must be'
                             ' divisible by the number of groups')
        W_shape = (self.out_channels, int(in_channels / self.groups), kh, kw)
        
        self.W.initialize(W_shape)
        # div scale
        self.scale = numpy.sqrt(numpy.mean(self.W.data**2))
        self.W.data = self.W.data / self.scale

    def __call__(self, x):
        """Applies the convolution layer.
        Args:
            x (~chainer.Variable): Input image.
        Returns:
            ~chainer.Variable: Output of the convolution.
        """
        if self.W.data is None:
            self._initialize_params(x.shape[1])
            
        y = self.scale*convolution_2d.convolution_2d(
            x, self.W, None, self.stride, self.pad, dilate=self.dilate,
            groups=self.groups)
        return bias(y,self.b)
    

class EqualizedLinear(link.Link):

    """Linear layer (a.k.a.\\  fully-connected layer).
    This is a link that wraps the :func:`~chainer.functions.linear` function,
    and holds a weight matrix ``W`` and optionally a bias vector ``b`` as
    parameters.
    If ``initialW`` is left to the default value of ``None``, the weight matrix
    ``W`` is initialized with i.i.d. Gaussian samples, each of which has zero
    mean and deviation :math:`\\sqrt{1/\\text{in_size}}`. The bias vector ``b``
    is of size ``out_size``. If the ``initial_bias`` is to left the default
    value of ``None``, each element is initialized as zero.  If the ``nobias``
    argument is set to ``True``, then this link does not hold a bias vector.
    Args:
        in_size (int or None): Dimension of input vectors. If unspecified or
            ``None``, parameter initialization will be deferred until the
            first forward data pass at which time the size will be determined.
        out_size (int): Dimension of output vectors. If only one value is
            passed for ``in_size`` and ``out_size``, that value will be used
            for the ``out_size`` dimension.
        nobias (bool): If ``True``, then this function does not use the bias.
        initialW (:ref:`initializer <initializer>`): Initializer to initialize
            the weight. When it is :class:`numpy.ndarray`,
            its ``ndim`` should be 2. If ``initialW`` is ``None``, then the
            weights are initialized with i.i.d. Gaussian samples, each of which
            has zero mean and deviation :math:`\\sqrt{1/\\text{in_size}}`.
        initial_bias (:ref:`initializer <initializer>`): Initializer to
            initialize the bias. If ``None``, the bias will be initialized to
            zero. When it is :class:`numpy.ndarray`, its ``ndim`` should be 1.
    .. seealso:: :func:`~chainer.functions.linear`
    Attributes:
        W (~chainer.Variable): Weight parameter.
        b (~chainer.Variable): Bias parameter.
    .. admonition:: Example
        There are several ways to make a Linear link.
        Define an input vector ``x`` as:
        >>> x = np.array([[0, 1, 2, 3, 4]], np.float32)
        1. Give the first two arguments explicitly:
            Those numbers are considered as the input size and the output size.
            >>> l = L.Linear(5, 10)
            >>> y = l(x)
            >>> y.shape
            (1, 10)
        2. Omit ``in_size`` (give the output size only as the first argument)
           or fill it with ``None``:
            In this case, the size of second axis of ``x`` is used as the
            input size. So the below two cases are the same.
            >>> l = L.Linear(10)
            >>> y = l(x)
            >>> y.shape
            (1, 10)
            >>> l = L.Linear(None, 10)
            >>> y = l(x)
            >>> y.shape
            (1, 10)
            When you omit the first argument, you need to specify the other
            subsequent arguments from ``nobias`` as keyword arguments. So the
            below two cases are the same.
            >>> l = L.Linear(None, 10, False, None, 0)
            >>> y = l(x)
            >>> y.shape
            (1, 10)
            >>> l = L.Linear(10, nobias=False, initialW=None, initial_bias=0)
            >>> y = l(x)
            >>> y.shape
            (1, 10)
    """

    def __init__(self, in_size, out_size=None, nobias=False,
                 initialW=None, initial_bias=None):
        super(EqualizedLinear, self).__init__()

        if out_size is None:
            in_size, out_size = None, in_size
        self.out_size = out_size

        with self.init_scope():
            W_initializer = initializers._get_initializer(initialW)
            self.W = variable.Parameter(W_initializer)
            if in_size is not None:
                self._initialize_params(in_size)

            if nobias:
                self.b = None
            else:
                if initial_bias is None:
                    initial_bias = 0
                bias_initializer = initializers._get_initializer(initial_bias)
                self.b = variable.Parameter(bias_initializer, out_size)

    def _initialize_params(self, in_size):
        self.W.initialize((self.out_size, in_size))
        # div scale
        self.scale = numpy.sqrt(numpy.mean(self.W.data**2))
        self.W.data = self.W.data / self.scale
        

    def __call__(self, x):
        """Applies the linear layer.
        Args:
            x (~chainer.Variable): Batch of input vectors.
        Returns:
            ~chainer.Variable: Output of the linear layer.
        """
        if self.W.data is None:
            in_size = functools.reduce(operator.mul, x.shape[1:], 1)
            self._initialize_params(in_size)
        #return linear.linear(x, self.W, self.b)
        y = self.scale * linear.linear(x, self.W, None)
        
        return bias(y,self.b)

def _pair(x):
    if hasattr(x, '__getitem__'):
        return x
    return x, x
        

