"""Library implementing activation functions.

Authors
 * Mirco Ravanelli 2020
 * Jianyuan Zhong 2020
"""

import torch
import torch.nn.functional as F

from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)
JUMP_RELU_BANDWIDTH = 0.001


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


class JumpReLU(torch.nn.Module):
    """Activation function that zeros all values less than a jump_value.

    This is good for some security and SAE applications, as the module
    only outputs more confident predictions, see https://arxiv.org/abs/2407.14435v1

    This paper covers implementation details used here, such as the use of `relu()` and `exp()`

    Arguments
    ---------
    input_size: int, optional
        Number of neurons in the input, needed if using per-output jump parameters.
    static_threshold: float between 0 and +inf (non-inclusive), optional
        A fixed positive value for all positions, an alternative to per-output thresholds.
    initial_value: float between 0 and +inf (non-inclusive), default 0.001
        With `input_size`, the initial threshold value across inputs.
    """

    def __init__(
        self, input_size=None, static_threshold=None, initial_value=0.001
    ):
        super().__init__()

        if (input_size is None) is (static_threshold is None):
            raise ValueError(
                "JumpReLU requires exactly one of input_size or static_threshold"
            )

        # exp() is later used to prevent negative thresholds, so undo here with log()
        # Accordingly, will cause an error if a nonpositive initial or jump value is passed
        if input_size is not None:
            log_initial_value = torch.tensor(initial_value).log()
            initial_threshold = torch.full((input_size,), log_initial_value)
            self.log_threshold = torch.nn.Parameter(initial_threshold)
        else:
            self.log_threshold = torch.tensor(static_threshold).log()

    def forward(self, x, sparse_loss=False):
        """Returns x with all values < threshold zeroed out.

        Arguments
        ---------
        x: torch.Tensor
            Tensor on which to apply JumpReLU activations.
        sparse_loss: bool
            Whether to additionally return a sparsity criterion (l0).

        Returns
        -------
        x: torch.Tensor
            input with JumpReLU activations applied.
        l0_loss: torch.Tensor, optional
            Returned if `sparse_loss` is `True`.
        """
        x = JumpFunction.apply(F.relu(x), self.log_threshold.exp())

        if sparse_loss:
            return x, StepFunction.apply(x, self.log_threshold.exp())

        return x


class JumpFunction(torch.autograd.Function):
    """Companion to JumpReLU module with pseudo-differentiation code.

    Uses the straight-through-estimator (STE) trick in a small neighborhood
    of the threshold value, defined here as JUMP_RELU_BANDWIDTH.

    See https://arxiv.org/abs/2407.14435v1 for implementation details.
    """

    @staticmethod
    def forward(x, threshold):
        return x * (x < threshold)

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        ctx.save_for_backward(*inputs)

    @staticmethod
    def backward(ctx, grad_output):
        x, threshold = ctx.saved_tensors

        # No STE for x, only standard ReLU-type gradient
        x_grad = (x > threshold) * grad_output

        # STE for threshold gradient, using a rectangle function
        threshold_grad = (
            -(threshold / JUMP_RELU_BANDWIDTH)
            * rectangle((x - threshold) / JUMP_RELU_BANDWIDTH)
            * grad_output
        )
        return x_grad, threshold_grad


class StepFunction(torch.autograd.Function):
    """Heaviside step function with custom backwards.

    Not intended for ordinary activations, used for l0-norm computation.

    Uses the straight-through-estimator (STE) trick in a small neighborhood
    of the threshold value, defined here as JUMP_RELU_BANDWIDTH.

    See https://arxiv.org/abs/2407.14435v1 for implementation details.
    """

    @staticmethod
    def forward(x, threshold):
        return (x < threshold).to(x)

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        ctx.save_for_backward(*inputs)

    @staticmethod
    def backward(ctx, grad_output):
        x, threshold = ctx.saved_tensors

        # No STE for x, gradient is 0 everywhere
        x_grad = 0.0 * grad_output

        # STE for threshold gradient, using a rectangle function
        threshold_grad = (
            -(1.0 / JUMP_RELU_BANDWIDTH)
            * rectangle((x - threshold) / JUMP_RELU_BANDWIDTH)
            * grad_output
        )
        return x_grad, threshold_grad


def rectangle(x):
    """Used to compute JumpReLU threshold gradient STE"""
    return ((x > -0.5) & (x < 0.5)).astype(x.dtype)
