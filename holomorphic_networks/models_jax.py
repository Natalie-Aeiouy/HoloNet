"""JAX-optimized complex-valued neural network models following best practices."""

from typing import List, Optional, Dict, Tuple, Callable, Any, NamedTuple
import jax
import jax.numpy as jnp
from jax import Array, random, lax
from functools import partial
import chex  # Add to requirements for runtime type checking

from .activations import get_activation, ACTIVATIONS
from .layers import ComplexLinear, ComplexLayerNorm


class MLPConfig(NamedTuple):
    """Immutable configuration for ComplexMLP to avoid issues with JIT."""
    layer_sizes: Tuple[int, ...]
    activation: str = 'h_elu'
    final_activation: Optional[str] = None
    use_bias: bool = True
    use_layer_norm: bool = False
    dropout_rate: float = 0.0
    dtype: Any = jnp.complex64


def create_mlp_params(config: MLPConfig, key: random.PRNGKey) -> Dict:
    """Pure function to initialize MLP parameters.
    
    This is a pure function that can be safely JIT-compiled.
    
    Args:
        config: MLP configuration
        key: Random key for initialization
        
    Returns:
        Dictionary containing all model parameters
    """
    num_layers = len(config.layer_sizes) - 1
    
    # Pre-allocate all keys to avoid repeated splitting in loop
    keys = random.split(key, 2 * num_layers if config.use_layer_norm else num_layers)
    key_idx = 0
    
    layers = []
    layer_norms = [] if config.use_layer_norm else None
    
    for i in range(num_layers):
        # Initialize linear layer
        linear = ComplexLinear(
            config.layer_sizes[i],
            config.layer_sizes[i + 1],
            use_bias=config.use_bias,
            dtype=config.dtype
        )
        layers.append(linear.init_params(keys[key_idx]))
        key_idx += 1
        
        # Initialize layer norm if needed (except for last layer)
        if config.use_layer_norm and i < num_layers - 1:
            ln = ComplexLayerNorm(
                config.layer_sizes[i + 1],
                dtype=config.dtype
            )
            layer_norms.append(ln.init_params(keys[key_idx]))
            key_idx += 1
    
    return {
        'layers': layers,
        'layer_norms': layer_norms
    }


@partial(jax.jit, static_argnames=['config', 'training'])
def mlp_forward(
    config: MLPConfig,
    params: Dict,
    x: Array,
    training: bool = False,
    key: Optional[random.PRNGKey] = None
) -> Tuple[Array, Dict]:
    """Pure function for MLP forward pass.
    
    This function is pure and can be safely JIT-compiled.
    Uses lax.fori_loop for better XLA compilation.
    
    Args:
        config: MLP configuration (static)
        params: Model parameters
        x: Input array of shape (batch, input_dim)
        training: Whether in training mode (static)
        key: Random key for dropout
        
    Returns:
        Tuple of (output, auxiliary_data)
    """
    num_layers = len(config.layer_sizes) - 1
    activation_fn = get_activation(config.activation)
    final_activation_fn = (
        get_activation(config.final_activation) 
        if config.final_activation else lambda x: x
    )
    
    # Initialize auxiliary data with fixed structure
    magnitudes_list = []
    activations_list = []
    
    # Handle dropout keys properly
    if training and config.dropout_rate > 0 and key is not None:
        dropout_keys = random.split(key, num_layers)
    else:
        dropout_keys = [None] * num_layers
    
    # Process through layers using functional approach
    for i in range(num_layers):
        # Linear transformation (pure function)
        w = params['layers'][i]['weight']
        x = jnp.dot(x, w.T)
        
        if config.use_bias:
            b = params['layers'][i]['bias']
            x = x + b
        
        # Layer normalization (except last layer)
        if config.use_layer_norm and params['layer_norms'] is not None and i < num_layers - 1:
            ln_params = params['layer_norms'][i]
            # Apply layer norm
            axes = tuple(range(-1, 0))
            mean = jnp.mean(x, axis=axes, keepdims=True)
            centered = x - mean
            variance = jnp.mean(jnp.abs(centered) ** 2, axis=axes, keepdims=True)
            normalized = centered / jnp.sqrt(variance + 1e-5)
            
            if 'gamma' in ln_params:
                normalized = ln_params['gamma'] * normalized + ln_params['beta']
            x = normalized
        
        # Activation (except last layer unless specified)
        if i < num_layers - 1:
            x = activation_fn(x)
        else:
            x = final_activation_fn(x)
        
        # Dropout (using JAX's functional approach)
        if training and config.dropout_rate > 0 and dropout_keys[i] is not None:
            keep_prob = 1 - config.dropout_rate
            keep_mask = random.bernoulli(dropout_keys[i], keep_prob, x.shape)
            x = lax.select(
                keep_mask,
                x / keep_prob,
                jnp.zeros_like(x)
            )
        
        # Track magnitude statistics (no mutation, create new data)
        magnitudes = jnp.abs(x)
        mag_stats = {
            'mean': jnp.mean(magnitudes),
            'std': jnp.std(magnitudes),
            'max': jnp.max(magnitudes),
            'min': jnp.min(magnitudes)
        }
        magnitudes_list.append(mag_stats)
        activations_list.append(x)
    
    aux_data = {
        'magnitudes': magnitudes_list,
        'activations': activations_list
    }
    
    return x, aux_data


@partial(jax.jit, static_argnames=['config'])
def mlp_apply(params: Dict, x: Array, config: MLPConfig) -> Array:
    """Simplified pure function for inference.
    
    Args:
        params: Model parameters
        x: Input array
        config: Model configuration
        
    Returns:
        Output array
    """
    output, _ = mlp_forward(config, params, x, training=False, key=None)
    return output


class ComplexMLPJAX:
    """JAX-optimized Complex MLP following best practices."""
    
    def __init__(
        self,
        layer_sizes: List[int],
        activation: str = 'h_elu',
        final_activation: Optional[str] = None,
        use_bias: bool = True,
        use_layer_norm: bool = False,
        dropout_rate: float = 0.0,
        dtype=jnp.complex64
    ):
        """Initialize Complex MLP with immutable config."""
        # Convert to immutable config
        self.config = MLPConfig(
            layer_sizes=tuple(layer_sizes),
            activation=activation,
            final_activation=final_activation,
            use_bias=use_bias,
            use_layer_norm=use_layer_norm,
            dropout_rate=dropout_rate,
            dtype=dtype
        )
        
        # Create JIT-compiled functions with proper static arguments
        self._forward_fn = partial(mlp_forward, self.config)
        self._apply_fn = partial(mlp_apply, config=self.config)
    
    def init_params(self, key: random.PRNGKey) -> Dict:
        """Initialize parameters using pure function."""
        return create_mlp_params(self.config, key)
    
    def forward(
        self,
        params: Dict,
        x: Array,
        training: bool = False,
        key: Optional[random.PRNGKey] = None
    ) -> Tuple[Array, Dict]:
        """Forward pass using JIT-compiled pure function."""
        return self._forward_fn(
            params=params,
            x=x,
            training=training,
            key=key
        )
    
    def apply(self, params: Dict, x: Array) -> Array:
        """Fast inference using optimized JIT function."""
        return self._apply_fn(params, x)
    
    def __call__(
        self,
        params: Dict,
        x: Array,
        training: bool = False,
        key: Optional[random.PRNGKey] = None
    ) -> Array:
        """Convenience method for forward pass."""
        output, _ = self.forward(params, x, training, key)
        return output


# Pure functional training step
@partial(jax.jit, static_argnames=['loss_fn', 'optimizer'])
def train_step_pure(
    params: Dict,
    opt_state: Any,
    batch: Tuple[Array, Array],
    loss_fn: Callable,
    optimizer: Any,
    key: Optional[random.PRNGKey] = None
) -> Tuple[Dict, Any, Dict]:
    """Pure training step function following JAX best practices.
    
    This function:
    - Is pure (no side effects)
    - Properly handles random keys
    - Returns new state instead of mutating
    
    Args:
        params: Model parameters
        opt_state: Optimizer state
        batch: Training batch (inputs, targets)
        loss_fn: Loss function
        optimizer: Optax optimizer
        key: Random key for dropout
        
    Returns:
        Tuple of (new_params, new_opt_state, metrics)
    """
    inputs, targets = batch
    
    # Define loss function with proper key handling
    def loss_with_key(p):
        # Split key for potential dropout
        if key is not None:
            key_dropout, key_metric = random.split(key)
        else:
            key_dropout = key_metric = None
            
        predictions = loss_fn(p, inputs, key_dropout)
        loss = jnp.mean(jnp.abs(predictions - targets) ** 2)
        return loss, key_metric
    
    # Compute gradients
    (loss_value, _), grads = jax.value_and_grad(loss_with_key, has_aux=True)(params)
    
    # Update parameters (returns new state, doesn't mutate)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    
    # Compute metrics
    metrics = {
        'loss': loss_value,
        'grad_norm': optax.global_norm(grads),
        'param_norm': optax.global_norm(new_params),
        'update_norm': optax.global_norm(updates)
    }
    
    return new_params, new_opt_state, metrics


# Scan-based training for efficient compilation
@partial(jax.jit, static_argnames=['n_steps', 'model_apply_fn', 'optimizer'])
def train_epoch_scan(
    params: Dict,
    opt_state: Any,
    data: Tuple[Array, Array],
    n_steps: int,
    batch_size: int,
    model_apply_fn: Callable,
    optimizer: Any,
    key: random.PRNGKey
) -> Tuple[Dict, Any, Dict]:
    """Train for one epoch using lax.scan for efficiency.
    
    Uses lax.scan instead of Python loops for better XLA compilation.
    
    Args:
        params: Initial parameters
        opt_state: Initial optimizer state
        data: Full dataset (inputs, targets)
        n_steps: Number of training steps
        batch_size: Batch size
        model_apply_fn: Model forward function
        optimizer: Optimizer
        key: Random key
        
    Returns:
        Tuple of (final_params, final_opt_state, metrics)
    """
    inputs, targets = data
    n_samples = inputs.shape[0]
    
    # Create batch indices
    indices = jnp.arange(n_samples)
    
    def scan_fn(carry, step_key):
        params, opt_state, indices = carry
        
        # Sample batch
        batch_key, dropout_key = random.split(step_key)
        batch_indices = random.choice(batch_key, indices, (batch_size,), replace=False)
        batch_inputs = inputs[batch_indices]
        batch_targets = targets[batch_indices]
        
        # Training step
        new_params, new_opt_state, metrics = train_step_pure(
            params, opt_state, (batch_inputs, batch_targets),
            model_apply_fn, optimizer, dropout_key
        )
        
        return (new_params, new_opt_state, indices), metrics
    
    # Generate keys for all steps
    step_keys = random.split(key, n_steps)
    
    # Run training loop with scan
    (final_params, final_opt_state, _), all_metrics = lax.scan(
        scan_fn, (params, opt_state, indices), step_keys
    )
    
    # Aggregate metrics
    avg_metrics = jax.tree.map(lambda x: jnp.mean(x), all_metrics)
    
    return final_params, final_opt_state, avg_metrics


# Utility functions for safe array operations
@jax.jit
def safe_normalize(x: Array, eps: float = 1e-8) -> Array:
    """Safely normalize complex array avoiding division by zero.
    
    Args:
        x: Complex array
        eps: Small epsilon for numerical stability
        
    Returns:
        Normalized array
    """
    magnitude = jnp.abs(x)
    # Use lax.select instead of jnp.where for better XLA compilation
    safe_magnitude = lax.select(
        magnitude > eps,
        magnitude,
        jnp.ones_like(magnitude)
    )
    return x / safe_magnitude


@jax.jit
def clip_gradients(grads: Any, max_norm: float) -> Any:
    """Clip gradients by global norm.
    
    Args:
        grads: Gradient tree
        max_norm: Maximum gradient norm
        
    Returns:
        Clipped gradients
    """
    grad_norm = optax.global_norm(grads)
    scale = lax.select(
        grad_norm > max_norm,
        max_norm / (grad_norm + 1e-8),
        1.0
    )
    return jax.tree.map(lambda g: g * scale, grads)


# Debug utilities that work with JIT
@jax.jit
def check_finite(x: Any, name: str = "tensor") -> Array:
    """Check if array contains finite values.
    
    Args:
        x: Array or pytree to check
        name: Name for debugging
        
    Returns:
        Boolean indicating if all values are finite
    """
    def is_finite(arr):
        if isinstance(arr, jnp.ndarray):
            return jnp.all(jnp.isfinite(arr))
        return True
    
    all_finite = jax.tree_util.tree_reduce(
        lambda a, b: a & b,
        jax.tree.map(is_finite, x)
    )
    
    # Use jax.debug.print for debugging with JIT
    jax.debug.print(
        f"{name} has non-finite values: {1 - all_finite}",
        ordered=True
    )
    
    return all_finite


# Proper random key management
class KeyManager:
    """Manages random keys properly for JAX.
    
    Ensures keys are never reused and properly split.
    """
    
    def __init__(self, seed: int = 42):
        """Initialize with seed."""
        self.key = random.PRNGKey(seed)
    
    def next_key(self) -> random.PRNGKey:
        """Get next key and update internal state."""
        self.key, subkey = random.split(self.key)
        return subkey
    
    def split(self, n: int) -> List[random.PRNGKey]:
        """Split into n keys."""
        self.key, *subkeys = random.split(self.key, n + 1)
        return subkeys


# Vectorized operations for batch processing
@jax.jit
def batch_mlp_forward(
    configs: List[MLPConfig],
    params_list: List[Dict],
    x: Array
) -> Array:
    """Apply multiple MLPs in parallel using vmap.
    
    Args:
        configs: List of MLP configurations
        params_list: List of parameters for each MLP
        x: Input array
        
    Returns:
        Stacked outputs from all MLPs
    """
    # Use vmap for parallel execution
    vmapped_forward = jax.vmap(
        lambda p, c: mlp_apply(p, x, c),
        in_axes=(0, None)
    )
    
    # Stack parameters
    params_stacked = jax.tree.map(
        lambda *ps: jnp.stack(ps),
        *params_list
    )
    
    return vmapped_forward(params_stacked, configs[0])


# Memory-efficient gradient accumulation
@partial(jax.jit, static_argnames=['n_accumulate'])
def accumulated_gradient_step(
    params: Dict,
    opt_state: Any,
    batches: List[Tuple[Array, Array]],
    loss_fn: Callable,
    optimizer: Any,
    n_accumulate: int
) -> Tuple[Dict, Any, float]:
    """Compute gradients over multiple batches for memory efficiency.
    
    Args:
        params: Model parameters
        opt_state: Optimizer state
        batches: List of batches to accumulate over
        loss_fn: Loss function
        optimizer: Optimizer
        n_accumulate: Number of batches to accumulate
        
    Returns:
        Updated parameters, optimizer state, and average loss
    """
    def compute_batch_grad(p, batch):
        inputs, targets = batch
        predictions = loss_fn(p, inputs, None)
        loss = jnp.mean(jnp.abs(predictions - targets) ** 2)
        return loss, jax.grad(lambda pr: loss)(p)
    
    # Accumulate gradients
    total_loss = 0.0
    accumulated_grads = jax.tree.map(jnp.zeros_like, params)
    
    for batch in batches[:n_accumulate]:
        loss, grads = compute_batch_grad(params, batch)
        total_loss += loss
        accumulated_grads = jax.tree.map(
            lambda a, g: a + g / n_accumulate,
            accumulated_grads, grads
        )
    
    # Apply accumulated gradients
    updates, new_opt_state = optimizer.update(accumulated_grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    
    return new_params, new_opt_state, total_loss / n_accumulate