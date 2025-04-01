"""The SpeechBrain implementation of various pre-trained model adapters e.g.
LoRA, Houlsby

Authors
 * Titouan Parcollet 2024
 * Peter Plantinga 2024
"""

import warnings
from fnmatch import fnmatch

import torch
import torch.nn as nn

from speechbrain.nnet.activations import Swish
from speechbrain.nnet.losses import mse_loss
from speechbrain.utils import checkpoints

MHA_WARNING = """
Torch's native multi-head attention is not adaptable since it accesses layer
weights directly to pass to highly optimized fused kernels. We are excluding
all native Torch MHA layers from the list of layers to adapt.
"""


@checkpoints.register_checkpoint_hooks
class AdaptedModel(nn.Module):
    """Given any torch model, e.g. asr_brain.modules.Transformer, and an adapter
    class, e.g. HoulsbyAdapter, this class will replace the target layers
    with this new adapter class (while preserving the parameters).

    Arguments
    ---------
    model_to_adapt: nn.Module
        The base PyTorch model to add adapters to.
    adapter_class: class
        An (uninitialized) adapter of this SpeechBrain library.
    all_linear: bool
        Whether to add the adapter to all linear layers (default: False)
    all_conv: bool
        Whether to add the adapter to all conv layers (default: False)
    target_layers: list of str
        A list of module names in the given model that should be replaced.
        Supports Unix shell-style wildcards `(*, ?, [seq], [!seq])` with `fnmatch`.
    unfrozen_layers: list of str
        List of layers to be unfrozen during training.
        Supports Unix shell-style wildcards `(*, ?, [seq], [!seq])` with `fnmatch`.
    adapter_kwargs: dict
        Ensemble of parameters that should be given to the adapter.
    manual_adapter_insertion: bool
        The default value (`False`) leads to the adapters being inserted at
        the time of initialization. However, in some cases, it is preferable
        to wait to insert the adapters, e.g. when pretrained parameters need to
        be loaded. In this case, one can set this to `True` and call
        `insert_adapters` manually after the parameters have been loaded.

    Example
    -------
    >>> from collections import OrderedDict
    >>> model = torch.nn.Sequential(
    ...   OrderedDict([
    ...     ("layer1", torch.nn.Linear(10, 20)),
    ...     ("layer2", torch.nn.Linear(20, 20)),
    ...     ("layer3", torch.nn.Linear(20, 10)),
    ...   ])
    ... )
    >>> lora_model = AdaptedModel(
    ...   model_to_adapt=model,
    ...   adapter_class=LoRA,
    ...   target_layers=["layer[13]"],
    ...   unfrozen_layers=["layer2"],
    ...   adapter_kwargs={"rank": 2},
    ... )
    >>> lora_model
    AdaptedModel(
      (adapted_model): Sequential(
        (layer1): LoRA(
          (pretrained_module): Linear(in_features=10, out_features=20, bias=True)
          (adapter_down_proj): Linear(in_features=10, out_features=2, bias=False)
          (adapter_up_proj): Linear(in_features=2, out_features=20, bias=False)
        )
        (layer2): Linear(in_features=20, out_features=20, bias=True)
        (layer3): LoRA(
          (pretrained_module): Linear(in_features=20, out_features=10, bias=True)
          (adapter_down_proj): Linear(in_features=20, out_features=2, bias=False)
          (adapter_up_proj): Linear(in_features=2, out_features=10, bias=False)
        )
      )
    )
    """

    def __init__(
        self,
        model_to_adapt: nn.Module,
        adapter_class: nn.Module,
        all_linear: bool = False,
        all_conv: bool = False,
        target_layers: list = [],
        unfrozen_layers: list = [],
        adapter_kwargs: dict = {},
        manual_adapter_insertion: bool = False,
    ):
        super().__init__()

        # Collect and freeze layers
        self.adapted_model = model_to_adapt
        self.adapter_class = adapter_class
        self.adapter_kwargs = adapter_kwargs
        for param in model_to_adapt.parameters():
            param.requires_grad = False

        # Iterate modules to create list of layers to adapt
        self.replace_layers = []
        for name, module in model_to_adapt.named_modules():
            if is_layer_adaptable(
                name, module, all_linear, all_conv, target_layers
            ):
                # Torch's MultiheadAttention is not adaptable due to an
                # optimized fused kernel, warn if we find this.
                parent_name = ".".join(name.split(".")[:-1])
                parent = model_to_adapt.get_submodule(parent_name)
                if isinstance(parent, torch.nn.MultiheadAttention):
                    warnings.warn(MHA_WARNING)
                else:
                    self.replace_layers.append(name)
            elif any(fnmatch(name, layer) for layer in unfrozen_layers):
                for param in module.parameters():
                    param.requires_grad = True

        if len(self.replace_layers) == 0:
            warnings.warn("No adaptable layers found")

        # Some cases require a delay in adapter insertion, e.g. using Pretrainer
        if not manual_adapter_insertion:
            self.insert_adapters()

    def insert_adapters(self):
        """If this is in `__init__` it conflicts with `Pretrainer`.
        Ensure this function is called exactly once before training.
        See ``__init__.manual_adapter_insertion``
        """
        for name in self.replace_layers:
            module = self.adapted_model.get_submodule(name)
            new_module = self.adapter_class(module, **self.adapter_kwargs)
            replace_module(self.adapted_model, name, new_module)

    def forward(self, *args, **kwargs):
        """Pass arguments to adapted model."""
        return self.adapted_model(*args, **kwargs)

    @checkpoints.mark_as_saver
    def saver(self, path):
        """Saves only the trainable parameters."""
        # NOTE: In order to preserve the gradient info, we have to prevent `state_dict` from detaching
        # all the parameters and buffers. The `keep_vars=True` does this, then we detach manually
        state_dict = {
            name: param.detach()
            for name, param in self.state_dict(keep_vars=True).items()
            if param.requires_grad
        }
        torch.save(state_dict, path)

    @checkpoints.mark_as_loader
    def loader(self, path, end_of_epoch):
        """Loads the base model plus trained params."""
        del end_of_epoch
        state_dict = torch.load(path, map_location="cpu")
        self.load_state_dict(state_dict, strict=False)

    @checkpoints.mark_as_transfer
    def parameter_transfer(self, path):
        """Avoids warnings due to only loading trained params."""
        self.loader(path, True)

    def __getattr__(self, item):
        """Override getattr to pass item accesses to pre-adapted model."""

        # Have to use super to get adapted model to avoid recursion
        model = super().__getattr__("adapted_model")
        if hasattr(model, item):
            return getattr(model, item)

        # Normal access
        return super().__getattr__(item)

    def encode(self, x):
        """Return the storage of any adapters with the capability."""

        # Enable storage for modules that support it
        for name in self.replace_layers:
            module = self.adapted_model.get_submodule(name)
            if hasattr(module, "enable_storage"):
                module.enable_storage()

        # Perform forward pass to store intermediate embeddings
        x = self.forward(x)

        # Collect embeddings from all modules that support it
        encoded_outputs = []
        for name in self.replace_layers:
            module = self.adapted_model.get_submodule(name)
            if hasattr(module, "get_storage"):
                encoded_outputs.append(module.get_storage())
            if hasattr(module, "disable_storage"):
                module.disable_storage()

        return torch.stack(encoded_outputs)


def is_layer_adaptable(name, module, all_linear, all_conv, target_layers):
    """Check if layer is among list of layers to be adapted.

    Arguments
    ---------
    name: str
        The name of the module to check.
    module: torch.nn.Module
        The module to check.
    all_linear: bool
        Whether all linear layers should be adapted.
    all_conv: bool
        Whether all conv layers should be adapted.
    target_layers: str or list of str
        See `add_adapters_to_model`

    Returns
    -------
    bool
        Whether the layer is to be adapted or not.
    """
    return (
        all_linear
        and isinstance(module, nn.Linear)
        or all_conv
        and isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d))
        or name
        and any(fnmatch(name, layer) for layer in target_layers)
    )


def replace_module(model: nn.Module, name: str, new_module: nn.Module):
    """Replace layer with a new module based on a parent assignation.
    This is used to replace layers with an Adapter layer wrapped around
    the original layer. Hence, old parameters are preserved and new ones are
    added.

    Arguments
    ---------
    model: nn.Module
        Model containing the module to be replaced.
    name: str
        Name of the target module to replace.
    new_module: nn.Module
        New module made of the old plus the new parameters.
    """

    # If the model is only one level deep, just use the model
    try:
        parent_name, target_name = name.rsplit(".", 1)
        parent_module = model.get_submodule(parent_name)
    except ValueError:
        parent_module = model
        target_name = name

    setattr(parent_module, target_name, new_module)


class HoulsbyAdapterLinear(nn.Module):
    """This class implements the Houlsby Adapter as described in:
    'Parameter-Efficient Transfer Learning for NLP'
    https://arxiv.org/abs/1902.00751

    Arguments
    ---------
    target_linear: nn.Module
        Module corresponding to the pretrained Linear that will be wrapped with
        this adapter.
    projection_size: int
        Size of the projection layer (usually smaller).
    activation: nn.Module
        The activation function. Default is Swish.
    bias: bool
        Whether to use biases in the linear projections.

    Example
    -------
    >>> import torch
    >>> x = torch.rand((8, 60, 64))
    >>> base_linear = nn.Linear(64, 64)
    >>> adapt = HoulsbyAdapterLinear(base_linear, 8)
    >>> output = adapt(x)
    >>> output.shape
    torch.Size([8, 60, 64])
    """

    def __init__(
        self,
        target_linear,
        projection_size,
        activation=Swish,
        bias=True,
    ):
        super().__init__()

        if not isinstance(target_linear, nn.Linear):
            raise ValueError(
                "HoulsbyLinear currently only supports linear layers, "
                f"but instead got {type(target_linear)}."
            )

        output_size = target_linear.weight.data.shape[0]
        device = target_linear.weight.device

        self.pretrained_linear = target_linear
        self.pretrained_linear.requires_grad = False
        self.adapter_down_proj = nn.Linear(
            output_size, projection_size, bias=bias, device=device
        )
        self.adapter_up_proj = nn.Linear(
            projection_size, output_size, bias=bias, device=device
        )
        self.activation = activation()

        if bias:
            self.adapter_down_proj.bias.data.fill_(0.0)
            self.adapter_up_proj.bias.data.fill_(0.0)

    def forward(self, x: torch.Tensor):
        """Applies the HoulsbyAdapter to an input tensor `x`.

        Arguments
        ---------
        x: torch.Tensor
            Input tensor to the adapter module. Shape: [B, Time, X]

        Returns
        -------
        The linear outputs
        """

        x_pretrained = self.pretrained_linear(x)

        return (
            self.adapter_up_proj(
                self.activation(self.adapter_down_proj(x_pretrained))
            )
            + x_pretrained
        )


class LoRA(nn.Module):
    """This class implements the LoRA Adapter as described in:
    'LoRA: Low-Rank Adaptation of Large Language Models'
    https://arxiv.org/abs/2106.09685

    Arguments
    ---------
    target_module: nn.Module
        Module corresponding to the pretrained layer that will be wrapped with
        this adapter. Works with nn.Linear and nn.Conv
    rank: int
        Size of the projection layer or rank (usually smaller).
    alpha : float
        Value used to control the scaling in LoRA. Default is one.

    Example
    -------
    >>> import torch
    >>> x = torch.rand((8, 60, 64))
    >>> base_linear = nn.Linear(64, 64)
    >>> adapt = LoRA(base_linear, 64, 4)
    >>> output = adapt(x)
    >>> output.shape
    torch.Size([8, 60, 64])
    """

    def __init__(self, target_module, rank=16, alpha=1.0):
        super().__init__()

        input_size = target_module.weight.data.shape[1]
        output_size = target_module.weight.data.shape[0]

        # Disable gradient for pretrained module
        self.pretrained_module = target_module
        for param in self.pretrained_module.parameters():
            param.requires_grad = False
        device = target_module.weight.device

        self.adapter_down_proj = nn.Linear(
            input_size, rank, bias=False, device=device
        )
        self.adapter_up_proj = nn.Linear(
            rank, output_size, bias=False, device=device
        )
        self.adapter_up_proj.weight.data.fill_(0.0)

        self.scaling = alpha / rank

    def forward(self, x: torch.Tensor):
        """Applies the LoRA Adapter.

        Arguments
        ---------
        x: torch.Tensor
            Input tensor to the adapter module.

        Returns
        -------
        The linear outputs
        """
        x_pretrained = self.pretrained_module(x)
        x_lora = self.adapter_up_proj(self.adapter_down_proj(x)) * self.scaling

        return x_pretrained + x_lora


class MaskTemp:
    """Keeps track of temperature for masking

    Arguments
    ---------
    steps: int
        Number of steps to reduce temperature over
    start_temp: float
        Starting temperature, usually > 1.0
    stop_temp: float
        Final temperature, usually between 0. and 1.
    """

    def __init__(self, steps, start_temp=2.0, stop_temp=0.1):
        assert start_temp > stop_temp
        assert steps > 0
        self.steps = steps
        self.stop_temp = stop_temp
        self.temperature = start_temp
        self.temperature_step_size = (start_temp - stop_temp) / steps

    def update_temp(self):
        """Update temperature one step while ensuring we don't go below stop_temp"""
        if self.temperature > self.stop_temp:
            self.temperature = max(
                self.temperature - self.temperature_step_size, self.stop_temp
            )

    def __call__(self):
        return self.temperature


class SparseAutoEncoder(nn.Module):
    """Adds a sparse auto-encoder (SAE) layer after a pretrained module.

    Majority of initialization code comes from a reference implementation.
    Github repo: saprmarks/dictionary_learning
    Relevant file: dictionary_learning/dictionary.py

    Arguments
    ---------
    target_module: torch.nn.Module
        A pretrained module for inserting the SAE layer.
    dict_size: int
        The number of neurons in the SAE layer, usually larger than the module output.
    module_out_size: int, optional
        This class can automatically determine the size of the module output only
        in the case where it is a simple linear layer, otherwise, this must be
        specified as an argument to the class.
    activation_fn: torch.nn.Module, default torch.nn.ReLU
        The class to use for activation, usually a ReLU-family function
        that creates sparse outputs by zeroing some outputs.
        Compatible with speechbrain.nnet.activations.JumpReLU().
        Also accepts "mask" which adds mask estimation parameters and loss.
    fidelity_loss_fn: loss fn, default speechbrain.nnet.losses.mse_loss
        A loss function that takes predicions, targets for autoencoder loss.
    storing_activations: bool, default False
        Whether to start storing activations and losses. Can be changed later with
        `enable_storage()` and `disable_storage()`.
    sparse_loss_fn: str, default "L1"
        Options are "L1", "L0", but ensure activation supports the argument
        "sparse_loss" if you would like to compute the "L0" loss.
    mask_temperature: MaskTemp
        The starting, stopping, and number of decay steps for the temperature
        used in the sigmoid for each mask item.


    Example
    -------
    >>> test_input = torch.rand(10)
    >>> module = torch.nn.Linear(10, 20)
    >>> sae = SparseAutoEncoder(module, dict_size=100)
    >>> sae.enable_storage()
    >>> sae(test_input).size()
    torch.Size([20])
    >>> sae.sparse_loss.size()
    torch.Size([100])
    >>> torch.allclose(sae.encode(test_input), sae.get_activations())
    True
    """

    def __init__(
        self,
        target_module,
        dict_size,
        module_out_size=None,
        activation_fn=torch.nn.ReLU(),
        fidelity_loss_fn=mse_loss,
        storing_activations=False,
        sparse_loss_fn="L0",
        mask_temperature=MaskTemp(start_temp=2.0, stop_temp=0.1, steps=1000),
    ):
        super().__init__()

        self.fidelity_loss_fn = fidelity_loss_fn

        # Ensure module is frozen and collect info
        self.pretrained_module = target_module
        for param in target_module.parameters():
            param.requires_grad = False
            device = param.device
        if module_out_size is None:
            module_out_size = target_module.weight.data.shape[0]

        # Initialize the machinery for caching the activations and loss
        self.fidelity_loss = None
        self.sparse_loss = None
        self.sparse_activations = None
        self.storing_activations = storing_activations
        self.activation_fn = activation_fn
        self.sparse_loss_fn = sparse_loss_fn
        self.mask_temperature = mask_temperature

        # Initialize the parameters according to reference
        self.dict_size = dict_size
        self.W_enc = nn.Parameter(
            torch.empty(module_out_size, dict_size, device=device)
        )
        self.b_enc = nn.Parameter(torch.zeros(dict_size, device=device))

        if self.activation_fn == "mask":
            self.W_mask = nn.Parameter(
                torch.nn.init.kaiming_uniform_(
                    torch.empty(module_out_size, dict_size, device=device)
                )
            )
            self.b_mask = nn.Parameter(torch.zeros(dict_size, device=device))

        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(dict_size, module_out_size, device=device)
            )
        )
        self.b_dec = nn.Parameter(torch.zeros(module_out_size, device=device))
        self.W_dec.data = self.W_dec / self.W_dec.norm(dim=1, keepdim=True)
        self.W_enc.data = self.W_dec.data.clone().T

    def forward(self, x):
        """Full forward pass, encode then decode for training."""

        if self.training:
            activations, sparse_loss = self.encode(x, sparse_loss=True)
            predictions = self.decode(activations)

            if self.storing_activations:
                self.sparse_activations = activations
                self.sparse_loss = sparse_loss
                targets = self.pretrained_module(x).detach()
                self.fidelity_loss = self.fidelity_loss_fn(predictions, targets)
        else:
            activations = self.encode(x, sparse_loss=False)
            predictions = self.decode(activations)

            if self.storing_activations:
                self.sparse_activations = activations

        return predictions

    def encode(self, x, sparse_loss=False):
        """Returns the sparse dictionary activations used for interpretability.

        Basically uses either the loss defined by the activation module
        (i.e. L0 loss when used with JumpReLU loss function) or the sparse loss in:
        https://transformer-circuits.pub/2024/april-update/index.html#training-saes

        Arguments
        ---------
        x: torch.Tensor
            Input tensor to the adapter model.
        sparse_loss: bool
            Whether to additionally return the sparse loss. In the case of
            JumpReLU this is L0 defined in the module itself, otherwise
            returns L1 loss.

        Returns
        -------
        activations: torch.Tensor
            The dictionary activations.
        sparse_loss: torch.Tensor (conditional)
            The sparsity loss defined by activation or L1.
        """
        if hasattr(self.pretrained_module, "attn_pooling_w"):
            out = self.pretrained_module.attn_pooling_w(x).squeeze(-1).float()
            out = torch.nn.functional.softmax(out, dim=-1).unsqueeze(-1)
            self.attention_scores = out

        inputs = self.pretrained_module(x)
        pre_activations = inputs @ self.W_enc + self.b_enc
        self.pre_activations = pre_activations

        if self.activation_fn == "mask":
            if self.training:
                t = 1 / self.mask_temperature()
                mask = (t * (inputs @ self.W_mask + self.b_mask)).sigmoid()
                # self.diversity_loss = mask.mean(dim=0) * mask.mean(dim=0).log()
                self.diversity_loss = torch.relu(mask.mean(dim=0) - 0.5)
            else:
                mask = (inputs @ self.W_mask + self.b_mask) > 0
            activations = mask.float() * pre_activations
            if sparse_loss:
                loss = (mask.mean(dim=0) * self.W_dec.norm(dim=1)).mean()
                return activations, loss
            return activations
        # If the L0 loss is not supported by the activation_fn, this will fail
        elif sparse_loss and self.sparse_loss_fn == "L0":
            return self.activation_fn(pre_activations, sparse_loss=sparse_loss)
        else:
            activations = self.activation_fn(pre_activations)
            if sparse_loss:
                sparse_loss_term = activations.abs().mean(dim=0)
                sparse_loss_term *= self.W_dec.norm(dim=1)
                return activations, sparse_loss_term.mean()
            return activations

    def decode(self, encoding):
        """Returns the decoded activations for use by the next layer."""
        return encoding @ self.W_dec + self.b_dec

    def update_temperature(self):
        """Forward mask temp update to correct class."""
        self.mask_temperature.update_temp()

    def enable_storage(self):
        """Turn on the storage of activations during the forward pass."""
        self.storing_activations = True

    def disable_storage(self):
        """Turn off the storage of activations during the forward pass."""
        self.storing_activations = False

    def get_activations(self, pre_activations=False):
        """Return the stored activations from the most recent forward pass."""
        return self.sparse_activations

    def get_sparse_loss(self):
        """Return the stored sparse loss from the most recent forward pass."""
        return self.sparse_loss

    def get_fidelity_loss(self):
        """Return the stored fidelity loss from the most recent forward pass."""
        return self.fidelity_loss
