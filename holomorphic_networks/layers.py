"""Complex-valued neural network layers for JAX."""

from typing import Optional, Tuple, Callable, Union
import jax
import jax.numpy as jnp
from jax import Array, random
from jax.nn.initializers import Initializer
import numpy as np


def complex_variance_scaling(
    scale: float = 1.0,
    mode: str = "fan_avg",
    distribution: str = "truncated_normal"
) -> Initializer:
    """Variance scaling initializer for complex weights.
    
    Extends He/Glorot initialization to complex domain.
    
    Args:
        scale: Scaling factor
        mode: One of "fan_in", "fan_out", "fan_avg"
        distribution: "truncated_normal" or "uniform"
        
    Returns:
        Initializer function for complex weights
    """
    def init(key: random.PRNGKey, shape: Tuple[int, ...], dtype=jnp.complex64) -> Array:
        fan_in = shape[0] if len(shape) >= 1 else 1
        fan_out = shape[1] if len(shape) >= 2 else 1
        
        if mode == "fan_in":
            denominator = fan_in
        elif mode == "fan_out":
            denominator = fan_out
        else:  # fan_avg
            denominator = (fan_in + fan_out) / 2
        
        variance = scale / denominator
        
        if distribution == "truncated_normal":
            stddev = jnp.sqrt(variance / 2)  # Divide by 2 for complex
            key_real, key_imag = random.split(key)
            real_part = random.truncated_normal(key_real, -2, 2, shape) * stddev
            imag_part = random.truncated_normal(key_imag, -2, 2, shape) * stddev
        else:  # uniform
            limit = jnp.sqrt(3 * variance / 2)  # Divide by 2 for complex
            key_real, key_imag = random.split(key)
            real_part = random.uniform(key_real, shape, minval=-limit, maxval=limit)
            imag_part = random.uniform(key_imag, shape, minval=-limit, maxval=limit)
        
        return real_part + 1j * imag_part
    
    return init


def complex_glorot_uniform() -> Initializer:
    """Glorot uniform initializer for complex weights."""
    return complex_variance_scaling(scale=1.0, mode="fan_avg", distribution="uniform")


def complex_glorot_normal() -> Initializer:
    """Glorot normal initializer for complex weights."""
    return complex_variance_scaling(scale=1.0, mode="fan_avg", distribution="truncated_normal")


def complex_he_uniform() -> Initializer:
    """He uniform initializer for complex weights."""
    return complex_variance_scaling(scale=2.0, mode="fan_in", distribution="uniform")


def complex_he_normal() -> Initializer:
    """He normal initializer for complex weights."""
    return complex_variance_scaling(scale=2.0, mode="fan_in", distribution="truncated_normal")


class ComplexLinear:
    """Complex-valued linear layer.
    
    Implements: y = Wx + b where W, x, b are complex.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias: bool = True,
        w_init: Optional[Initializer] = None,
        b_init: Optional[Initializer] = None,
        dtype=jnp.complex64
    ):
        """Initialize complex linear layer.
        
        Args:
            in_features: Number of input features
            out_features: Number of output features
            use_bias: Whether to use bias term
            w_init: Weight initializer (default: complex Glorot normal)
            b_init: Bias initializer (default: zeros)
            dtype: Data type for weights (complex64 or complex128)
        """
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.dtype = dtype
        
        self.w_init = w_init or complex_glorot_normal()
        self.b_init = b_init or (lambda key, shape, dtype: jnp.zeros(shape, dtype=dtype))
    
    def init_params(self, key: random.PRNGKey) -> dict:
        """Initialize layer parameters.
        
        Args:
            key: Random key for initialization
            
        Returns:
            Dictionary with 'weight' and optionally 'bias' parameters
        """
        key_w, key_b = random.split(key)
        params = {}
        
        params['weight'] = self.w_init(
            key_w, (self.out_features, self.in_features), self.dtype
        )
        
        if self.use_bias:
            params['bias'] = self.b_init(key_b, (self.out_features,), self.dtype)
        
        return params
    
    def __call__(self, params: dict, x: Array) -> Array:
        """Apply complex linear transformation.
        
        Args:
            params: Layer parameters dictionary
            x: Input array of shape (..., in_features)
            
        Returns:
            Output array of shape (..., out_features)
        """
        # Complex matrix multiplication
        y = jnp.dot(x, params['weight'].T)
        
        if self.use_bias:
            y = y + params['bias']
        
        return y


class ComplexConv2D:
    """Complex-valued 2D convolutional layer."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[str, int, Tuple[int, int]] = 'SAME',
        use_bias: bool = True,
        w_init: Optional[Initializer] = None,
        b_init: Optional[Initializer] = None,
        dtype=jnp.complex64
    ):
        """Initialize complex convolutional layer.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of convolution kernel
            stride: Stride of convolution
            padding: Padding mode ('SAME', 'VALID', or explicit padding)
            use_bias: Whether to use bias term
            w_init: Weight initializer
            b_init: Bias initializer
            dtype: Data type for weights
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
            
        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride
            
        self.padding = padding
        self.use_bias = use_bias
        self.dtype = dtype
        
        self.w_init = w_init or complex_glorot_normal()
        self.b_init = b_init or (lambda key, shape, dtype: jnp.zeros(shape, dtype=dtype))
    
    def init_params(self, key: random.PRNGKey) -> dict:
        """Initialize layer parameters.
        
        Args:
            key: Random key for initialization
            
        Returns:
            Dictionary with 'weight' and optionally 'bias' parameters
        """
        key_w, key_b = random.split(key)
        params = {}
        
        # Weight shape: (out_channels, in_channels, height, width)
        weight_shape = (
            self.out_channels,
            self.in_channels,
            self.kernel_size[0],
            self.kernel_size[1]
        )
        
        params['weight'] = self.w_init(key_w, weight_shape, self.dtype)
        
        if self.use_bias:
            params['bias'] = self.b_init(key_b, (self.out_channels,), self.dtype)
        
        return params
    
    def __call__(self, params: dict, x: Array) -> Array:
        """Apply complex 2D convolution.
        
        Args:
            params: Layer parameters dictionary
            x: Input array of shape (batch, height, width, channels)
            
        Returns:
            Output array after convolution
        """
        # JAX expects weights in shape (height, width, in_channels, out_channels)
        weight = params['weight'].transpose(2, 3, 1, 0)
        
        # Complex convolution using JAX's conv_general_dilated
        y = jax.lax.conv_general_dilated(
            x,
            weight,
            window_strides=self.stride,
            padding=self.padding,
            dimension_numbers=('NHWC', 'HWIO', 'NHWC')
        )
        
        if self.use_bias:
            # Broadcast bias across spatial dimensions
            bias = params['bias'].reshape(1, 1, 1, -1)
            y = y + bias
        
        return y


class ComplexLayerNorm:
    """Complex-valued layer normalization.
    
    Normalizes complex features while preserving phase information.
    """
    
    def __init__(
        self,
        normalized_shape: Union[int, Tuple[int, ...]],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        dtype=jnp.complex64
    ):
        """Initialize complex layer normalization.
        
        Args:
            normalized_shape: Shape of features to normalize
            eps: Small constant for numerical stability
            elementwise_affine: Whether to use learnable affine parameters
            dtype: Data type for parameters
        """
        if isinstance(normalized_shape, int):
            self.normalized_shape = (normalized_shape,)
        else:
            self.normalized_shape = normalized_shape
            
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.dtype = dtype
    
    def init_params(self, key: random.PRNGKey) -> dict:
        """Initialize layer parameters.
        
        Args:
            key: Random key (unused, but kept for consistency)
            
        Returns:
            Dictionary with 'gamma' and 'beta' parameters if affine
        """
        params = {}
        
        if self.elementwise_affine:
            # Initialize gamma to 1+0j and beta to 0+0j
            params['gamma'] = jnp.ones(self.normalized_shape, dtype=self.dtype)
            params['beta'] = jnp.zeros(self.normalized_shape, dtype=self.dtype)
        
        return params
    
    def __call__(self, params: dict, x: Array) -> Array:
        """Apply complex layer normalization.
        
        Args:
            params: Layer parameters dictionary
            x: Input array
            
        Returns:
            Normalized complex array
        """
        # Calculate mean and variance of complex numbers
        # For complex numbers, we use the complex mean and variance
        axes = tuple(range(-len(self.normalized_shape), 0))
        
        mean = jnp.mean(x, axis=axes, keepdims=True)
        
        # Complex variance: E[|x - mean|^2]
        centered = x - mean
        variance = jnp.mean(jnp.abs(centered) ** 2, axis=axes, keepdims=True)
        
        # Normalize
        normalized = centered / jnp.sqrt(variance + self.eps)
        
        if self.elementwise_affine:
            normalized = params['gamma'] * normalized + params['beta']
        
        return normalized


class ComplexBatchNorm:
    """Complex-valued batch normalization.
    
    Extends batch normalization to complex domain.
    """
    
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        dtype=jnp.complex64
    ):
        """Initialize complex batch normalization.
        
        Args:
            num_features: Number of features (channels)
            eps: Small constant for numerical stability
            momentum: Momentum for running statistics
            affine: Whether to use learnable affine parameters
            track_running_stats: Whether to track running mean/variance
            dtype: Data type for parameters
        """
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.dtype = dtype
    
    def init_params(self, key: random.PRNGKey) -> dict:
        """Initialize layer parameters and statistics.
        
        Args:
            key: Random key (unused, but kept for consistency)
            
        Returns:
            Dictionary with parameters and running statistics
        """
        params = {}
        
        if self.affine:
            params['gamma'] = jnp.ones((self.num_features,), dtype=self.dtype)
            params['beta'] = jnp.zeros((self.num_features,), dtype=self.dtype)
        
        if self.track_running_stats:
            params['running_mean'] = jnp.zeros((self.num_features,), dtype=self.dtype)
            params['running_var'] = jnp.ones((self.num_features,), dtype=jnp.float32)
            params['num_batches_tracked'] = jnp.array(0, dtype=jnp.int32)
        
        return params
    
    def __call__(
        self,
        params: dict,
        x: Array,
        training: bool = True
    ) -> Tuple[Array, dict]:
        """Apply complex batch normalization.
        
        Args:
            params: Layer parameters dictionary
            x: Input array of shape (batch, ..., num_features)
            training: Whether in training mode
            
        Returns:
            Tuple of (normalized array, updated parameters)
        """
        # Get batch statistics
        axes = tuple(range(len(x.shape) - 1))  # All axes except last
        
        if training:
            batch_mean = jnp.mean(x, axis=axes)
            centered = x - batch_mean.reshape(1, -1)
            batch_var = jnp.mean(jnp.abs(centered) ** 2, axis=axes)
            
            # Update running statistics
            if self.track_running_stats:
                new_params = params.copy()
                new_params['running_mean'] = (
                    (1 - self.momentum) * params['running_mean'] +
                    self.momentum * batch_mean
                )
                new_params['running_var'] = (
                    (1 - self.momentum) * params['running_var'] +
                    self.momentum * batch_var
                )
                new_params['num_batches_tracked'] = params['num_batches_tracked'] + 1
            else:
                new_params = params
            
            mean = batch_mean
            var = batch_var
        else:
            new_params = params
            if self.track_running_stats:
                mean = params['running_mean']
                var = params['running_var']
            else:
                mean = jnp.mean(x, axis=axes)
                centered = x - mean.reshape(1, -1)
                var = jnp.mean(jnp.abs(centered) ** 2, axis=axes)
        
        # Normalize
        x_normalized = (x - mean.reshape(1, -1)) / jnp.sqrt(var.reshape(1, -1) + self.eps)
        
        if self.affine:
            x_normalized = (
                params['gamma'].reshape(1, -1) * x_normalized +
                params['beta'].reshape(1, -1)
            )
        
        return x_normalized, new_params


def create_mlp_params(
    key: random.PRNGKey,
    layer_sizes: list,
    activation: str = 'h_elu',
    use_bias: bool = True,
    dtype=jnp.complex64
) -> dict:
    """Create parameters for a complex MLP.
    
    Args:
        key: Random key for initialization
        layer_sizes: List of layer sizes [input, hidden1, ..., output]
        activation: Name of activation function
        use_bias: Whether to use bias in linear layers
        dtype: Data type for parameters
        
    Returns:
        Dictionary with all MLP parameters
    """
    params = {'layers': [], 'activation': activation}
    
    for i in range(len(layer_sizes) - 1):
        key, subkey = random.split(key)
        layer = ComplexLinear(
            layer_sizes[i],
            layer_sizes[i + 1],
            use_bias=use_bias,
            dtype=dtype
        )
        params['layers'].append(layer.init_params(subkey))
    
    return params