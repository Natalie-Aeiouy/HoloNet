# HoloNet Testing Guide 🧪

This guide will help you test and validate the HoloNet implementation step by step.

## 📋 Prerequisites

### 1. Activate Virtual Environment
```bash
cd /mnt/c/OntoDyn/HoloNet
source nvenv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

If you encounter issues, install core dependencies first:
```bash
pip install jax jaxlib numpy
pip install optax matplotlib
```

## 🚀 Quick Start Testing

### Option 1: Run Quick Start Script (Recommended First)
```bash
python quick_start.py
```

This will:
- Test all core components
- Train a simple model
- Compare backprop methods
- Show performance metrics

Expected output:
- ✅ All imports work
- ✅ Model trains and loss decreases
- ✅ Predictions are reasonable
- ✅ JAX JIT compilation shows speedup

### Option 2: Run Comprehensive Test Suite
```bash
python test_setup.py
```

This runs 9 systematic tests:
1. Basic imports
2. Package imports
3. Activation functions
4. Model forward pass
5. JAX optimization
6. Training
7. Backprop methods
8. Magnitude analysis
9. Visualization

## 🔍 Step-by-Step Testing

### Step 1: Verify JAX Installation
```python
python -c "import jax; print(f'JAX version: {jax.__version__}')"
python -c "import jax; print(f'Devices: {jax.devices()}')"
```

### Step 2: Test Imports
```python
python -c "from holomorphic_networks import holomorphic_elu, ComplexMLP; print('✅ Imports work!')"
```

### Step 3: Test Activation Functions
```python
python -c "
import jax.numpy as jnp
from holomorphic_networks.activations import holomorphic_elu
z = jnp.array([1.0 + 1.0j])
result = holomorphic_elu(z)
print(f'H-ELU({z[0]}) = {result[0]}')
"
```

### Step 4: Test Model Creation
```python
python -c "
from holomorphic_networks.models import ComplexMLP
from jax import random
model = ComplexMLP([2, 32, 1], activation='h_elu')
params = model.init_params(random.PRNGKey(42))
print('✅ Model created and initialized')
"
```

## 🧪 Unit Tests with Pytest

### Run All Tests
```bash
pytest tests/ -v
```

### Run Specific Test
```bash
pytest tests/test_activations.py -v
```

### With Coverage
```bash
pytest --cov=holomorphic_networks tests/
```

## 📊 Performance Testing

### Test JAX JIT Compilation
```python
from holomorphic_networks.models_jax import ComplexMLPJAX
import jax.numpy as jnp
from jax import random
import time

model = ComplexMLPJAX([10, 128, 128, 10])
params = model.init_params(random.PRNGKey(42))
x = random.normal(random.PRNGKey(0), (1000, 10)) + 1j * random.normal(random.PRNGKey(1), (1000, 10))

# First call (compilation)
start = time.time()
_ = model.apply(params, x)
print(f"First call: {time.time() - start:.3f}s")

# Second call (cached)
start = time.time()
_ = model.apply(params, x)
print(f"Second call: {time.time() - start:.3f}s")
```

### Test Backpropagation Methods
```python
from holomorphic_networks.experiments.backprop_comparison import run_backprop_comparison_experiment

results = run_backprop_comparison_experiment(
    task='polynomial',
    activation='h_elu',
    n_epochs=100
)
```

## 🔧 Troubleshooting

### Issue: ImportError
```bash
# Solution 1: Ensure you're in the right directory
cd /mnt/c/OntoDyn/HoloNet

# Solution 2: Add to Python path
export PYTHONPATH="${PYTHONPATH}:/mnt/c/OntoDyn/HoloNet"

# Solution 3: Install in development mode
pip install -e .
```

### Issue: JAX Not Using GPU
```python
# Check available devices
import jax
print(jax.devices())

# Force CPU if GPU issues
import os
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
```

### Issue: Memory Errors
```python
# Use smaller batch sizes
from holomorphic_networks.training import TrainingConfig
config = TrainingConfig(batch_size=16)  # Reduce from 32

# Enable memory-efficient gradients
from holomorphic_networks.jax_utils import checkpoint_layer
# Use checkpointing for large models
```

### Issue: NaN/Inf in Training
```python
# Enable NaN checking
from holomorphic_networks.jax_utils import check_nan_inf

# In your training loop
has_issues, stats = check_nan_inf(grads, "gradients")
if has_issues:
    print(f"NaN/Inf detected: {stats}")
    
# Use gradient clipping
from holomorphic_networks.training import TrainingConfig
config = TrainingConfig(gradient_clip=1.0)
```

## 📈 Expected Results

### Training z² Function
- Initial loss: ~1.0-10.0
- Final loss (100 epochs): <0.01
- Training time: <10 seconds

### Backprop Method Comparison
- JAX AutoDiff: Best performance (baseline)
- Wirtinger: Similar to JAX (theoretically equivalent)
- Split: 10-50% worse than JAX
- Naive Real: Often fails to converge

### Magnitude Drift (5-layer network)
- H-ELU: Drift ratio 0.5-2.0 (good preservation)
- H-Swish: Drift ratio 0.5-2.0
- Tanh: Drift ratio 0.01-0.1 (vanishing)
- CReLU: Drift ratio 1-10 (depends on init)

## 🎯 Quick Validation Checklist

Run these commands in order:

```bash
# 1. Basic test
python -c "import jax; print('JAX OK')"

# 2. Package test
python -c "from holomorphic_networks import ComplexMLP; print('Package OK')"

# 3. Quick functionality test
python quick_start.py

# 4. Full test suite
python test_setup.py

# 5. Unit tests (if pytest installed)
pytest tests/test_activations.py -v
```

If all pass, HoloNet is ready to use! 🎉

## 📚 Next Steps

1. **Explore Examples**:
   ```bash
   python holomorphic_networks/experiments/synthetic.py
   ```

2. **Try Custom Models**:
   ```python
   from holomorphic_networks import ComplexMLP
   model = ComplexMLP([input_dim, 64, 64, output_dim], activation='h_swish')
   ```

3. **Experiment with Backprop**:
   ```python
   model.train(data, backprop_method='wirtinger')
   ```

4. **Analyze Your Networks**:
   ```python
   from holomorphic_networks.analysis import comprehensive_analysis
   results = comprehensive_analysis(model_fn, params, data)
   ```

## 🆘 Getting Help

- Check `JAX_BEST_PRACTICES.md` for optimization tips
- Review `CLAUDE.md` for project-specific guidance
- Look at test files for usage examples
- Check the docstrings in the source code

Happy testing! 🚀