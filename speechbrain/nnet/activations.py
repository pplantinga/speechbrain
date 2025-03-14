"""Library implementing activation functions.

Authors
 * Mirco Ravanelli 2020
 * Jianyuan Zhong 2020
"""

import torch
import torch.nn.functional as F

from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


class Softmax(torch.nn.Module):
    """Computes the softmax of a 2d, 3d, or 4d input tensor.

    Arguments
    ---------
    apply_log : bool
        Whether to apply the log function before softmax.
    dim : int
        If the dimension where softmax is applied.
    reshape: bool
        whether to apply reshaping (true by default)
    dtype: torch.dtype
        dtype of the output tensor

    Example
    -------
    >>> classifier = Softmax()
    >>> inputs = torch.rand(10, 50, 40)
    >>> output = classifier(inputs)
    >>> output.shape
    torch.Size([10, 50, 40])
    """

    def __init__(
        self, apply_log=False, dim=-1, reshape=True, dtype=torch.float32
    ):
        super().__init__()

        if apply_log:
            self.act = F.log_softmax
        else:
            self.act = F.softmax

        self.dim = dim
        self.reshape = reshape
        self.dtype = dtype

    def forward(self, x):
        """Returns the softmax of the input tensor.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        x_act : torch.Tensor
            The softmax outputs.
        """
        # Reshaping the tensors
        dims = x.shape

        if self.reshape:
            if len(dims) == 3:
                x = x.reshape(dims[0] * dims[1], dims[2])

            if len(dims) == 4:
                x = x.reshape(dims[0] * dims[1], dims[2], dims[3])

        x_act = self.act(x, dim=self.dim, dtype=self.dtype)

        # Retrieving the original shape format
        if self.reshape:
            if len(dims) == 3:
                x_act = x_act.reshape(dims[0], dims[1], dims[2])

            if len(dims) == 4:
                x_act = x_act.reshape(dims[0], dims[1], dims[2], dims[3])

        return x_act


class GumbelSoftmax(torch.nn.Module):
    """Samples from the Gumbel-Softmax distribution and optionally discretizes.

    Reference: https://arxiv.org/abs/1611.00712, https://arxiv.org/abs/1611.01144

    Arguments
    ---------
    tau: float
        non-negative scalar temperature
    hard: bool
        if True, the returned samples will be discretized as one-hot vectors, but will be differentiated as if it is the soft sample in autograd
    apply_log: bool
        if True, returns the log of the softmax outputs.

    Example
    -------
    >>> x = torch.randn((8, 40, 120))
    >>> act = GumbelSoftmax(0.8, True)
    >>> x = act(x)
    """

    def __init__(self, tau, hard=False, apply_log=False):
        super().__init__()
        self.tau = tau
        self.hard = hard
        self.apply_log = apply_log

    def forward(self, x):
        """Returns the Gumbel softmax of the input tensor.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        The Gumbel softmax output.
        """
        if self.apply_log:
            return torch.log(F.gumbel_softmax(x, tau=self.tau, hard=self.hard))
        return F.gumbel_softmax(x, tau=self.tau, hard=self.hard)


class Swish(torch.nn.Module):
    """The class implements the Swish activation function from
    https://arxiv.org/pdf/2005.03191.pdf

    given input x. Swish(x) = x / (1 + exp(beta * x))

    Arguments
    ---------
    beta: float
        Beta value.

    Example
    -------
    >>> x = torch.randn((8, 40, 120))
    >>> act = Swish()
    >>> x = act(x)
    """

    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta
        self.silu = torch.nn.SiLU()

    def forward(self, x):
        """Returns the Swished input tensor.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        The swished output.
        """
        if self.beta != 1:  # slow path
            x = x * self.beta

        return self.silu(x)


class BatchTopK(torch.nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, x):
        # Record batch size and flatten
        flattened_x = x.flatten()
        topk = flattened_x.topk(self.k * x.size(0), sorted=False, dim=-1)

        # Return original shape with zeros except in topk positions
        return (
            torch.zeros_like(flattened_x)
            .scatter_(-1, topk.indices, topk.values)
            .reshape(x.shape)
        )


class JumpReLU(torch.nn.Module):
    """Activation function that zeros all values less than a jump_value.

    This is good for some security and SAE applications, as the module
    only outputs more confident predictions, see https://arxiv.org/abs/2407.14435v1
    This paper covers implementation details, but we use this repo as reference:

    https://github.com/saprmarks/dictionary_learning

    Arguments
    ---------
    input_size: int, optional
        Number of neurons in the input to establish per-input thresholds
    static_threshold: float, optional
        Static global threshold value, used instead of per-input thresholds
    initial_threshold: non-negative float, default 0.001
        Initial threshold value to apply across inputs
    grad_bandwidth: float, default 0.001
        The width of the gradient around the threshold.
    """

    def __init__(
        self,
        input_size=None,
        static_threshold=None,
        initial_threshold=0.001,
        grad_bandwidth=0.001,
    ):
        super().__init__()

        if (input_size is None) is (static_threshold is None):
            raise ValueError(
                "Must specify exactly one of input_size and static_threshold"
            )

        self.bandwidth = grad_bandwidth
        if input_size is not None:
            initial_threshold = torch.full((input_size,), initial_threshold)
            self.threshold = torch.nn.Parameter(initial_threshold)
        else:
            self.threshold = torch.tensor(static_threshold)

    def forward(self, x, sparse_loss=False):
        """Returns x with all values < threshold zeroed out.

        Arguments
        ---------
        x: torch.Tensor
            Tensor on which to apply JumpReLU activations
        sparse_loss: bool
            Whether to additionally return a sparsity criterion (l0)

        Returns
        -------
        x: torch.Tensor
            input with JumpReLU activations applied
        l0_loss: torch.Tensor (conditional)
            Returned if `sparse_loss` is `True`
        """
        x = JumpReLUFunction.apply(F.relu(x), self.threshold, self.bandwidth)

        if sparse_loss:
            x = (x, StepFunction.apply(x, self.threshold, self.bandwidth))

        return x


class RectangleFunction(torch.autograd.Function):
    """Rectangle / delta function used in JumpReLU and StepFunction

    See https://arxiv.org/abs/2407.14435v1 for implementation details.

    Implementation from: saprmarks/dictionary_learning github
    Implementation file: dictionary_learning/trainers/jumprelu.py
    """

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return ((x > -0.5) & (x < 0.5)).float()

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[(x <= -0.5) | (x >= 0.5)] = 0
        return grad_input


class JumpReLUFunction(torch.autograd.Function):
    """Companion to JumpReLU module with pseudo-differentiation code.

    Uses the straight-through-estimator (STE) trick in a small neighborhood
    of the threshold value, defined here as JUMP_RELU_BANDWIDTH.

    See https://arxiv.org/abs/2407.14435v1 for implementation details.

    Implementation from: saprmarks/dictionary_learning github
    Implementation file: dictionary_learning/trainers/jumprelu.py
    """

    @staticmethod
    def forward(ctx, x, threshold, bandwidth):
        ctx.save_for_backward(x, threshold, torch.tensor(bandwidth))
        return x * (x > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        x, threshold, bandwidth_tensor = ctx.saved_tensors
        bandwidth = bandwidth_tensor.item()
        x_grad = (x > threshold).float() * grad_output
        threshold_grad = (
            -(threshold / bandwidth)
            * RectangleFunction.apply((x - threshold) / bandwidth)
            * grad_output
        )
        return x_grad, threshold_grad, None  # None for bandwidth


class StepFunction(torch.autograd.Function):
    """Heaviside step function with custom backwards.

    Not intended for ordinary activations, used for l0-loss computation.

    Uses the straight-through-estimator (STE) trick in a small neighborhood
    of the threshold value, the "bandwidth".

    See https://arxiv.org/abs/2407.14435v1 for implementation details.

    Implementation from: saprmarks/dictionary_learning github
    Implementation file: dictionary_learning/trainers/jumprelu.py
    """

    @staticmethod
    def forward(ctx, x, threshold, bandwidth):
        ctx.save_for_backward(x, threshold, torch.tensor(bandwidth))
        return (x > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        x, threshold, bandwidth_tensor = ctx.saved_tensors
        bandwidth = bandwidth_tensor.item()
        x_grad = torch.zeros_like(x)
        threshold_grad = (
            -(1.0 / bandwidth)
            * RectangleFunction.apply((x - threshold) / bandwidth)
            * grad_output
        )
        return x_grad, threshold_grad, None  # None for bandwidth
