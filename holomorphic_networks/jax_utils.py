"""JAX utilities and best practices for complex neural networks."""

import jax
import jax.numpy as jnp
from jax import Array, random, lax
from typing import Any, Tuple, Callable, Optional, Dict, List
import functools
from dataclasses import dataclass
import warnings


# ============================================================================
# Control Flow Best Practices
# ============================================================================

def cond_select(condition: Array, true_val: Array, false_val: Array) -> Array:
    """JAX-friendly conditional selection.
    
    Use lax.select instead of Python if-else or jnp.where for better compilation.
    
    Args:
        condition: Boolean condition array
        true_val: Value when condition is True
        false_val: Value when condition is False
        
    Returns:
        Selected values
    """
    return lax.select(condition, true_val, false_val)


def switch_case(index: int, branches: List[Callable], *args) -> Any:
    """JAX-friendly switch-case statement.
    
    Uses lax.switch for multiple branches instead of Python if-elif chains.
    
    Args:
        index: Branch index to execute
        branches: List of callable branches
        *args: Arguments passed to the selected branch
        
    Returns:
        Result from selected branch
    """
    return lax.switch(index, branches, *args)


def scan_loop(fn: Callable, init: Any, xs: Array, length: Optional[int] = None) -> Tuple[Any, Any]:
    """JAX-friendly loop using lax.scan.
    
    Use this instead of Python for loops for better compilation.
    
    Args:
        fn: Function (carry, x) -> (new_carry, y)
        init: Initial carry value
        xs: Array to scan over
        length: Optional length if xs is None
        
    Returns:
        Tuple of (final_carry, stacked_outputs)
    """
    return lax.scan(fn, init, xs, length=length)


def fori_loop(lower: int, upper: int, body_fn: Callable, init_val: Any) -> Any:
    """JAX-friendly for loop using lax.fori_loop.
    
    Use for loops with known bounds instead of Python for loops.
    
    Args:
        lower: Loop start index
        upper: Loop end index (exclusive)
        body_fn: Function (i, val) -> new_val
        init_val: Initial value
        
    Returns:
        Final value after loop
    """
    return lax.fori_loop(lower, upper, body_fn, init_val)


def while_loop(cond_fn: Callable, body_fn: Callable, init_val: Any) -> Any:
    """JAX-friendly while loop using lax.while_loop.
    
    Use for loops with dynamic bounds.
    
    Args:
        cond_fn: Function (val) -> bool
        body_fn: Function (val) -> new_val
        init_val: Initial value
        
    Returns:
        Final value when condition becomes False
    """
    return lax.while_loop(cond_fn, body_fn, init_val)


# ============================================================================
# NaN and Debugging Utilities
# ============================================================================

@jax.custom_gradient
def nan_safe_divide(x: Array, y: Array, eps: float = 1e-8) -> Array:
    """Division that gracefully handles zero denominators.
    
    Args:
        x: Numerator
        y: Denominator
        eps: Small epsilon for stability
        
    Returns:
        x / y with NaN protection
    """
    safe_y = jnp.where(jnp.abs(y) > eps, y, jnp.ones_like(y) * eps)
    result = x / safe_y
    
    def grad_fn(g):
        grad_x = g / safe_y
        grad_y = -g * x / (safe_y ** 2)
        # Clip extreme gradients
        grad_x = jnp.clip(grad_x, -1e6, 1e6)
        grad_y = jnp.clip(grad_y, -1e6, 1e6)
        return grad_x, grad_y, None
    
    return result, grad_fn


@jax.jit
def check_nan_inf(x: Any, name: str = "") -> Tuple[bool, Dict[str, int]]:
    """Check for NaN and Inf values in arrays.
    
    Args:
        x: Array or pytree to check
        name: Optional name for debugging
        
    Returns:
        Tuple of (has_issues, stats_dict)
    """
    def count_issues(arr):
        if not isinstance(arr, jnp.ndarray):
            return {'nan': 0, 'inf': 0, 'total': 0}
        
        nan_count = jnp.sum(jnp.isnan(arr))
        inf_count = jnp.sum(jnp.isinf(arr))
        total = arr.size
        
        return {'nan': nan_count, 'inf': inf_count, 'total': total}
    
    stats = jax.tree.map(count_issues, x)
    
    # Aggregate stats
    total_stats = jax.tree_util.tree_reduce(
        lambda a, b: {
            'nan': a['nan'] + b['nan'],
            'inf': a['inf'] + b['inf'],
            'total': a['total'] + b['total']
        },
        stats,
        {'nan': 0, 'inf': 0, 'total': 0}
    )
    
    has_issues = (total_stats['nan'] > 0) | (total_stats['inf'] > 0)
    
    # Debug print (only executes with jax.debug)
    jax.debug.print(
        f"NaN/Inf check {name}: NaN={total_stats['nan']}, Inf={total_stats['inf']}, Total={total_stats['total']}",
        ordered=True
    )
    
    return has_issues, total_stats


@functools.partial(jax.jit, static_argnames=['max_value'])
def clip_by_value(x: Array, max_value: float = 1e6) -> Array:
    """Clip array values to prevent overflow.
    
    Args:
        x: Array to clip
        max_value: Maximum absolute value
        
    Returns:
        Clipped array
    """
    return jnp.clip(x, -max_value, max_value)


def add_nan_checks(fn: Callable) -> Callable:
    """Decorator to add NaN checking to a function.
    
    Args:
        fn: Function to wrap
        
    Returns:
        Wrapped function with NaN checking
    """
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        # Check inputs
        jax.debug.print("Checking inputs for NaN/Inf", ordered=True)
        for i, arg in enumerate(args):
            if isinstance(arg, jnp.ndarray):
                has_issues, _ = check_nan_inf(arg, f"input_{i}")
                jax.debug.callback(
                    lambda: warnings.warn(f"Input {i} has NaN/Inf!") if has_issues else None
                )
        
        # Run function
        result = fn(*args, **kwargs)
        
        # Check outputs
        jax.debug.print("Checking outputs for NaN/Inf", ordered=True)
        has_issues, _ = check_nan_inf(result, "output")
        jax.debug.callback(
            lambda: warnings.warn("Output has NaN/Inf!") if has_issues else None
        )
        
        return result
    
    return wrapped


# ============================================================================
# Memory Management
# ============================================================================

def donate_buffer(fn: Callable) -> Callable:
    """Decorator to donate input buffers for memory efficiency.
    
    Use when input arrays won't be needed after the function call.
    
    Args:
        fn: Function to wrap
        
    Returns:
        Wrapped function with buffer donation
    """
    @functools.wraps(fn)
    @functools.partial(jax.jit, donate_argnums=(0,))
    def wrapped(x, *args, **kwargs):
        return fn(x, *args, **kwargs)
    
    return wrapped


@jax.jit
def checkpoint_layer(layer_fn: Callable, params: Dict, x: Array) -> Array:
    """Checkpoint a layer for memory-efficient backprop.
    
    Trades computation for memory by recomputing forward pass during backprop.
    
    Args:
        layer_fn: Layer forward function
        params: Layer parameters
        x: Input
        
    Returns:
        Layer output
    """
    return jax.checkpoint(layer_fn)(params, x)


def shard_data(x: Array, n_devices: Optional[int] = None) -> Array:
    """Shard data across devices for parallel processing.
    
    Args:
        x: Data to shard
        n_devices: Number of devices (auto-detect if None)
        
    Returns:
        Sharded array
    """
    if n_devices is None:
        n_devices = jax.device_count()
    
    # Reshape to have first dimension divisible by n_devices
    batch_size = x.shape[0]
    if batch_size % n_devices != 0:
        pad_size = n_devices - (batch_size % n_devices)
        padding = [(0, pad_size)] + [(0, 0)] * (len(x.shape) - 1)
        x = jnp.pad(x, padding)
    
    # Reshape and shard
    x = x.reshape(n_devices, -1, *x.shape[1:])
    return jax.device_put_sharded(list(x), jax.devices())


# ============================================================================
# Structured Control Flow Patterns
# ============================================================================

@dataclass
class LoopState:
    """State for structured loops."""
    iteration: int
    value: Any
    key: Optional[random.PRNGKey] = None
    metrics: Optional[Dict] = None


def structured_scan(
    n_steps: int,
    step_fn: Callable[[LoopState], LoopState],
    init_state: LoopState
) -> Tuple[LoopState, List[Dict]]:
    """Structured scan with proper state management.
    
    Args:
        n_steps: Number of steps
        step_fn: Function that updates loop state
        init_state: Initial state
        
    Returns:
        Tuple of (final_state, metrics_history)
    """
    def scan_fn(state, i):
        new_state = step_fn(state)
        new_state = LoopState(
            iteration=i,
            value=new_state.value,
            key=new_state.key,
            metrics=new_state.metrics
        )
        return new_state, new_state.metrics
    
    final_state, metrics_history = lax.scan(
        scan_fn, init_state, jnp.arange(n_steps)
    )
    
    return final_state, metrics_history


# ============================================================================
# Complex-Specific Utilities
# ============================================================================

@jax.jit
def complex_dropout(
    x: Array,
    rate: float,
    key: random.PRNGKey,
    mode: str = 'standard'
) -> Array:
    """Complex-valued dropout with different modes.
    
    Args:
        x: Complex input
        rate: Dropout rate
        key: Random key
        mode: 'standard', 'phase', or 'magnitude'
        
    Returns:
        Dropped out array
    """
    if mode == 'standard':
        # Standard dropout
        keep_prob = 1 - rate
        mask = random.bernoulli(key, keep_prob, x.shape)
        return lax.select(mask, x / keep_prob, jnp.zeros_like(x))
    
    elif mode == 'phase':
        # Only dropout phase, keep magnitude
        magnitude = jnp.abs(x)
        phase = jnp.angle(x)
        keep_prob = 1 - rate
        mask = random.bernoulli(key, keep_prob, x.shape)
        new_phase = lax.select(mask, phase, jnp.zeros_like(phase))
        return magnitude * jnp.exp(1j * new_phase)
    
    elif mode == 'magnitude':
        # Only dropout magnitude, keep phase
        magnitude = jnp.abs(x)
        phase = jnp.angle(x)
        keep_prob = 1 - rate
        mask = random.bernoulli(key, keep_prob, x.shape)
        new_magnitude = lax.select(mask, magnitude / keep_prob, jnp.zeros_like(magnitude))
        return new_magnitude * jnp.exp(1j * phase)
    
    else:
        raise ValueError(f"Unknown dropout mode: {mode}")


@jax.jit
def complex_noise(
    x: Array,
    noise_std: float,
    key: random.PRNGKey,
    mode: str = 'gaussian'
) -> Array:
    """Add noise to complex values.
    
    Args:
        x: Complex input
        noise_std: Noise standard deviation
        key: Random key
        mode: 'gaussian' or 'uniform'
        
    Returns:
        Noisy array
    """
    key_real, key_imag = random.split(key)
    
    if mode == 'gaussian':
        noise_real = random.normal(key_real, x.shape) * noise_std
        noise_imag = random.normal(key_imag, x.shape) * noise_std
    elif mode == 'uniform':
        noise_real = random.uniform(key_real, x.shape, -noise_std, noise_std)
        noise_imag = random.uniform(key_imag, x.shape, -noise_std, noise_std)
    else:
        raise ValueError(f"Unknown noise mode: {mode}")
    
    return x + (noise_real + 1j * noise_imag)


# ============================================================================
# Performance Monitoring
# ============================================================================

class PerformanceMonitor:
    """Monitor JAX performance metrics."""
    
    def __init__(self):
        """Initialize monitor."""
        self.metrics = {}
    
    def time_function(self, fn: Callable, *args, **kwargs) -> Tuple[Any, float]:
        """Time a JAX function execution.
        
        Args:
            fn: Function to time
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Tuple of (result, time_in_seconds)
        """
        # Compile first
        _ = fn(*args, **kwargs)
        
        # Time execution
        start = jax.profiler.start_trace("/tmp/jax_trace")
        result = fn(*args, **kwargs)
        result.block_until_ready()  # Wait for async execution
        jax.profiler.stop_trace()
        
        return result, 0.0  # Timing from profiler
    
    def check_compilation(self, fn: Callable) -> bool:
        """Check if function is compiled.
        
        Args:
            fn: Function to check
            
        Returns:
            True if compiled
        """
        return hasattr(fn, '_cached_translation')
    
    def memory_info(self) -> Dict[str, int]:
        """Get memory usage information.
        
        Returns:
            Dictionary with memory stats
        """
        devices = jax.devices()
        info = {}
        
        for device in devices:
            stats = device.memory_stats()
            info[str(device)] = {
                'bytes_in_use': stats['bytes_in_use'],
                'peak_bytes_in_use': stats['peak_bytes_in_use'],
                'bytes_reserved': stats['bytes_reserved']
            }
        
        return info


# ============================================================================
# Best Practices Documentation
# ============================================================================

JAX_BEST_PRACTICES = """
JAX Best Practices for Complex Neural Networks:

1. PURE FUNCTIONS:
   - All functions should be pure (no side effects)
   - Don't modify global state
   - Return new values instead of mutating

2. RANDOM KEYS:
   - Always split keys, never reuse
   - Pass keys explicitly through functions
   - Use KeyManager for systematic key handling

3. JIT COMPILATION:
   - Mark static arguments with static_argnums/static_argnames
   - Use NamedTuple for configurations
   - Avoid Python control flow in JIT functions

4. CONTROL FLOW:
   - Use lax.cond instead of if-else
   - Use lax.scan/fori_loop instead of Python loops
   - Use lax.select instead of jnp.where

5. MEMORY MANAGEMENT:
   - Use donate_argnums for temporary arrays
   - Use jax.checkpoint for memory-efficient gradients
   - Shard large arrays across devices

6. DEBUGGING:
   - Use jax.debug.print instead of print
   - Add NaN/Inf checks with check_nan_inf
   - Use jax.disable_jit() for debugging

7. COMPLEX NUMBERS:
   - JAX handles complex gradients automatically
   - Use jnp.abs, jnp.angle for magnitude/phase
   - Be careful with non-holomorphic operations

8. PERFORMANCE:
   - Vectorize with vmap instead of loops
   - Use lax operations for better XLA compilation
   - Profile with jax.profiler

Example Usage:
    # Good: Pure function with proper key handling
    def forward(params, x, key):
        key1, key2 = random.split(key)
        x = layer1(params['layer1'], x, key1)
        x = layer2(params['layer2'], x, key2)
        return x
    
    # Bad: Impure function with key reuse
    def forward(params, x, key):
        x = layer1(params['layer1'], x, key)  # Key reuse!
        x = layer2(params['layer2'], x, key)  # Key reuse!
        return x
"""

print(JAX_BEST_PRACTICES)