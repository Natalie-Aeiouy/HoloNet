# JAX Best Practices for HoloNet

This document outlines JAX best practices implemented in the HoloNet project to avoid common gotchas and ensure optimal performance.

## 🔑 Key Principles

### 1. Pure Functions
All functions must be pure (no side effects) for JAX compilation:

```python
# ✅ GOOD: Pure function
def forward(params, x):
    x = layer1(params['layer1'], x)
    x = layer2(params['layer2'], x)
    return x

# ❌ BAD: Mutating input
def forward(params, x):
    x[0] = 0  # Mutation!
    return x
```

### 2. Random Key Management
Never reuse random keys - always split:

```python
# ✅ GOOD: Proper key splitting
def dropout_layers(x, key):
    key1, key2, key3 = random.split(key, 3)
    x = dropout(x, key1)
    x = layer(x)
    x = dropout(x, key2)
    return x, key3

# ❌ BAD: Key reuse
def dropout_layers(x, key):
    x = dropout(x, key)  # Reuse!
    x = layer(x)
    x = dropout(x, key)  # Reuse!
    return x
```

### 3. JIT Compilation
Use static arguments correctly:

```python
# ✅ GOOD: Static arguments marked
@partial(jax.jit, static_argnames=['config', 'training'])
def forward(config, params, x, training=False):
    # config and training are static (known at compile time)
    if training:  # OK because training is static
        x = dropout(x)
    return x

# ❌ BAD: Dynamic values in static positions
@jax.jit
def forward(params, x, dropout_rate):
    if dropout_rate > 0:  # Will recompile for each dropout_rate!
        x = dropout(x, dropout_rate)
    return x
```

### 4. Control Flow
Use JAX control flow primitives:

```python
# ✅ GOOD: lax.cond for conditionals
x = lax.cond(
    condition,
    lambda x: x * 2,      # True branch
    lambda x: x + 1,      # False branch
    x
)

# ❌ BAD: Python if-else (breaks JIT)
if condition:
    x = x * 2
else:
    x = x + 1
```

```python
# ✅ GOOD: lax.scan for loops
def process_sequence(x, sequence):
    def step(carry, elem):
        new_carry = carry + elem
        return new_carry, new_carry
    
    final, all_carries = lax.scan(step, x, sequence)
    return final

# ❌ BAD: Python for loop
result = x
for elem in sequence:
    result = result + elem
```

### 5. Array Updates
Use functional updates instead of mutations:

```python
# ✅ GOOD: Functional update
array = array.at[index].set(value)
array = array.at[index].add(value)
array = array.at[index].multiply(value)

# ❌ BAD: In-place mutation
array[index] = value  # Doesn't work!
array[index] += value  # Doesn't work!
```

### 6. NaN/Inf Handling
Prevent and detect numerical issues:

```python
# ✅ GOOD: Safe operations
def safe_normalize(x, eps=1e-8):
    magnitude = jnp.abs(x)
    safe_mag = jnp.maximum(magnitude, eps)
    return x / safe_mag

# Check for issues
has_nan = jnp.any(jnp.isnan(x))
has_inf = jnp.any(jnp.isinf(x))

# ❌ BAD: Unsafe division
def normalize(x):
    return x / jnp.abs(x)  # Division by zero!
```

### 7. Memory Management
Optimize memory usage:

```python
# ✅ GOOD: Donate buffers when possible
@partial(jax.jit, donate_argnums=(0,))
def update_params(params, grads):
    # params can be donated since we don't need old values
    return params - 0.01 * grads

# ✅ GOOD: Checkpoint for memory efficiency
def deep_network(params, x):
    # Checkpoint intermediate layers
    x = jax.checkpoint(layer1)(params['layer1'], x)
    x = jax.checkpoint(layer2)(params['layer2'], x)
    return x
```

## 📦 HoloNet-Specific Implementations

### ComplexMLPJAX
The `models_jax.py` module provides a fully JAX-optimized implementation:

```python
from holomorphic_networks.models_jax import ComplexMLPJAX, MLPConfig

# Immutable configuration
config = MLPConfig(
    layer_sizes=(2, 64, 64, 1),
    activation='h_elu',
    dropout_rate=0.1
)

# Pure function initialization
model = ComplexMLPJAX(layer_sizes=[2, 64, 64, 1])
params = model.init_params(key)

# JIT-compiled forward pass
output, aux = model.forward(params, x, training=True, key=dropout_key)
```

### Key Manager
Systematic random key handling:

```python
from holomorphic_networks.jax_utils import KeyManager

keys = KeyManager(seed=42)

# Never worry about key reuse
key1 = keys.next_key()
key2 = keys.next_key()
key3, key4, key5 = keys.split(3)
```

### Structured Loops
Clean loop patterns:

```python
from holomorphic_networks.jax_utils import structured_scan, LoopState

def train_epoch(params, data, key):
    def step_fn(state):
        # Training step logic
        new_params = update(state.value, data[state.iteration])
        new_key, subkey = random.split(state.key)
        return LoopState(
            iteration=state.iteration + 1,
            value=new_params,
            key=new_key,
            metrics={'loss': compute_loss(new_params)}
        )
    
    init_state = LoopState(0, params, key, {})
    final_state, history = structured_scan(100, step_fn, init_state)
    return final_state.value, history
```

### Complex-Specific Operations
Safe complex number handling:

```python
from holomorphic_networks.jax_utils import (
    complex_dropout, 
    complex_noise,
    nan_safe_divide
)

# Complex-aware dropout
x = complex_dropout(x, rate=0.1, key=key, mode='magnitude')

# Safe complex division
z = nan_safe_divide(x, y, eps=1e-8)

# Add complex noise
x_noisy = complex_noise(x, noise_std=0.01, key=key)
```

## 🐛 Debugging

### Enable Debug Mode
```python
# Disable JIT for debugging
with jax.disable_jit():
    output = model.forward(params, x)

# Add debug prints (work with JIT)
jax.debug.print("x shape: {}", x.shape)
jax.debug.print("Has NaN: {}", jnp.any(jnp.isnan(x)))

# Check for numerical issues
from holomorphic_networks.jax_utils import check_nan_inf
has_issues, stats = check_nan_inf(params, "params")
```

### Performance Monitoring
```python
from holomorphic_networks.jax_utils import PerformanceMonitor

monitor = PerformanceMonitor()

# Time function execution
result, elapsed = monitor.time_function(model.forward, params, x)

# Check memory usage
memory_stats = monitor.memory_info()
print(f"Memory in use: {memory_stats}")
```

## ⚡ Performance Tips

1. **Vectorize with vmap**:
```python
# Process batch in parallel
batch_forward = jax.vmap(model.forward, in_axes=(None, 0))
outputs = batch_forward(params, batch_inputs)
```

2. **Use lax operations**:
```python
# Better XLA compilation
y = lax.select(condition, true_val, false_val)  # Not jnp.where
y = lax.clamp(min_val, x, max_val)  # Not jnp.clip
```

3. **Shard across devices**:
```python
from holomorphic_networks.jax_utils import shard_data
sharded_data = shard_data(large_array, n_devices=8)
```

## 🚫 Common Pitfalls to Avoid

### 1. Shape Issues
```python
# ❌ BAD: Dynamic shapes
def bad_network(x, n_hidden):
    return jnp.zeros((x.shape[0], n_hidden))  # Shape depends on data!

# ✅ GOOD: Static shapes
def good_network(x, config):
    return jnp.zeros((x.shape[0], config.hidden_size))  # config is static
```

### 2. Python Operations on Arrays
```python
# ❌ BAD: Python operations
if x > 0:  # x is array!
    y = x * 2

# ✅ GOOD: Array operations
y = lax.select(x > 0, x * 2, x)
```

### 3. Side Effects
```python
# ❌ BAD: Side effects
global_list = []
def bad_fn(x):
    global_list.append(x)  # Side effect!
    return x * 2

# ✅ GOOD: Pure function
def good_fn(x, history):
    new_history = history + [x]
    return x * 2, new_history
```

## 📚 Resources

- [JAX Documentation](https://jax.readthedocs.io/)
- [Common Gotchas in JAX](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html)
- [JAX 101](https://jax.readthedocs.io/en/latest/jax-101/index.html)
- [Thinking in JAX](https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html)

## 🎯 Quick Checklist

Before running your JAX code, check:

- [ ] All functions are pure (no mutations or side effects)
- [ ] Random keys are properly split, never reused
- [ ] Static arguments are marked in @jit decorators
- [ ] Using lax control flow instead of Python control flow
- [ ] Arrays are updated functionally, not mutated
- [ ] Numerical stability is ensured (no division by zero)
- [ ] Memory is managed efficiently (checkpointing, sharding)
- [ ] Debug utilities are in place for NaN/Inf detection