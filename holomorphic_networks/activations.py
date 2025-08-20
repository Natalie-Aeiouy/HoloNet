"""Holomorphic and baseline activation functions for complex-valued neural networks."""

from typing import Union
import jax
import jax.numpy as jnp
from jax import Array


def holomorphic_elu(z: Array, alpha: float = 1.0) -> Array:
    """Holomorphic Complex ELU activation function.
    
    For complex z = x + iy:
    f(z) = z if Re(z) > 0, else alpha * (exp(z) - 1)
    
    This is holomorphic everywhere and provides similar behavior to modrelu
    but without breaking holomorphicity by taking absolute values.
    
    Args:
        z: Complex input array
        alpha: Slope for negative real part (default: 1.0)
        
    Returns:
        Complex array with same shape as input
    """
    condition = z.real > 0
    return jnp.where(condition, z, alpha * (jnp.exp(z) - 1))


def holomorphic_swish(z: Array, beta: float = 1.0) -> Array:
    """Holomorphic Complex Swish activation: z * sigmoid(beta * z).
    
    f(z) = z * (1 / (1 + exp(-beta * z)))
    
    This is fully holomorphic and provides smooth, gated activation.
    
    Args:
        z: Complex input array
        beta: Scaling parameter for sigmoid (default: 1.0)
        
    Returns:
        Complex array with same shape as input
    """
    return z * jax.nn.sigmoid(beta * z)


def crelu(z: Array) -> Array:
    """Complex ReLU (CReLU) activation.
    
    f(z) = ReLU(Re(z)) + i*ReLU(Im(z))
    
    Applies ReLU separately to real and imaginary parts.
    Note: This breaks holomorphicity but is commonly used as baseline.
    
    Args:
        z: Complex input array
        
    Returns:
        Complex array with same shape as input
    """
    return jax.nn.relu(z.real) + 1j * jax.nn.relu(z.imag)


def modrelu(z: Array, bias: float = 0.0) -> Array:
    """Modular ReLU (ModReLU) activation.
    
    f(z) = ReLU(|z| + b) * (z/|z|) if |z| + b >= 0, else 0
    
    Uses magnitude-based gating with phase preservation.
    Note: This breaks holomorphicity due to magnitude operation.
    
    Args:
        z: Complex input array
        bias: Bias term added to magnitude (default: 0.0)
        
    Returns:
        Complex array with same shape as input
    """
    magnitude = jnp.abs(z)
    # Add small epsilon to avoid division by zero
    eps = 1e-8
    phase = z / (magnitude + eps)
    
    # Apply ReLU to magnitude + bias
    gated_magnitude = jax.nn.relu(magnitude + bias)
    
    return gated_magnitude * phase


def complex_tanh(z: Array) -> Array:
    """Complex hyperbolic tangent activation.
    
    f(z) = tanh(z)
    
    Standard complex tanh, holomorphic everywhere.
    
    Args:
        z: Complex input array
        
    Returns:
        Complex array with same shape as input
    """
    return jnp.tanh(z)


def zrelu(z: Array) -> Array:
    """zReLU activation (Guberman, 2016).
    
    f(z) = z if both Re(z) > 0 and Im(z) > 0, else 0
    
    Phase-sensitive ReLU that zeros output unless both components are positive.
    Note: This breaks holomorphicity.
    
    Args:
        z: Complex input array
        
    Returns:
        Complex array with same shape as input
    """
    condition = (z.real > 0) & (z.imag > 0)
    return jnp.where(condition, z, 0)


def cardioid(z: Array) -> Array:
    """Cardioid activation function.
    
    f(z) = 0.5 * (1 + cos(angle(z))) * z
    
    Phase-dependent scaling that creates cardioid-shaped activation region.
    Note: This breaks holomorphicity due to angle operation.
    
    Args:
        z: Complex input array
        
    Returns:
        Complex array with same shape as input
    """
    phase = jnp.angle(z)
    scale = 0.5 * (1 + jnp.cos(phase))
    return scale * z


def complex_sigmoid(z: Array) -> Array:
    """Complex sigmoid activation.
    
    f(z) = 1 / (1 + exp(-z))
    
    Holomorphic sigmoid function extended to complex domain.
    
    Args:
        z: Complex input array
        
    Returns:
        Complex array with same shape as input
    """
    return jax.nn.sigmoid(z)


def split_complex_relu(z: Array) -> Array:
    """Split complex ReLU with separate real/imaginary processing.
    
    Applies ReLU to real part and tanh to imaginary part.
    f(z) = ReLU(Re(z)) + i*tanh(Im(z))
    
    Note: This breaks holomorphicity.
    
    Args:
        z: Complex input array
        
    Returns:
        Complex array with same shape as input
    """
    return jax.nn.relu(z.real) + 1j * jnp.tanh(z.imag)


def magnitude_preserving_activation(z: Array, activation_fn=jnp.tanh) -> Array:
    """Apply activation while preserving magnitude.
    
    f(z) = |z| * activation_fn(z/|z|)
    
    Preserves the magnitude while applying activation to normalized input.
    Note: This breaks holomorphicity due to magnitude operations.
    
    Args:
        z: Complex input array
        activation_fn: Activation to apply to normalized input
        
    Returns:
        Complex array with same shape as input
    """
    magnitude = jnp.abs(z)
    eps = 1e-8
    normalized = z / (magnitude + eps)
    return magnitude * activation_fn(normalized)


# Activation function registry for easy access
ACTIVATIONS = {
    'h_elu': holomorphic_elu,
    'h_swish': holomorphic_swish,
    'crelu': crelu,
    'modrelu': modrelu,
    'tanh': complex_tanh,
    'zrelu': zrelu,
    'cardioid': cardioid,
    'sigmoid': complex_sigmoid,
    'split_relu': split_complex_relu
}


def get_activation(name: str):
    """Get activation function by name.
    
    Args:
        name: Name of activation function
        
    Returns:
        Activation function
        
    Raises:
        ValueError: If activation name not found
    """
    if name not in ACTIVATIONS:
        raise ValueError(f"Unknown activation: {name}. Available: {list(ACTIVATIONS.keys())}")
    return ACTIVATIONS[name]