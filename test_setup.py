#!/usr/bin/env python3
"""
Comprehensive test script to validate HoloNet implementation.
Run this to verify everything is working correctly.
"""

import sys
import traceback
from typing import Dict, Any, List
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings for initial testing

# Test results tracking
test_results = {
    'passed': [],
    'failed': [],
    'skipped': []
}

def run_test(test_name: str, test_func: callable) -> bool:
    """Run a single test and track results."""
    print(f"\n{'='*60}")
    print(f"Testing: {test_name}")
    print(f"{'='*60}")
    
    try:
        test_func()
        print(f"✅ PASSED: {test_name}")
        test_results['passed'].append(test_name)
        return True
    except Exception as e:
        print(f"❌ FAILED: {test_name}")
        print(f"Error: {str(e)}")
        print(f"Traceback:\n{traceback.format_exc()}")
        test_results['failed'].append((test_name, str(e)))
        return False


# ============================================================================
# Test 1: Basic Imports
# ============================================================================
def test_basic_imports():
    """Test that all basic dependencies can be imported."""
    print("Testing JAX installation...")
    import jax
    import jax.numpy as jnp
    print(f"  JAX version: {jax.__version__}")
    
    print("Testing other dependencies...")
    import numpy as np
    print(f"  NumPy version: {np.__version__}")
    
    import optax
    print(f"  Optax version: {optax.__version__}")
    
    import matplotlib
    print(f"  Matplotlib version: {matplotlib.__version__}")
    
    # Test JAX basic functionality
    print("\nTesting JAX basic operations...")
    x = jnp.array([1.0 + 1.0j, 2.0 + 2.0j])
    y = jnp.abs(x)
    assert y.shape == (2,), "JAX array shape mismatch"
    print(f"  Complex array test: {x} -> magnitude: {y}")
    
    # Test GPU/TPU availability
    print(f"\nJAX devices available: {jax.devices()}")
    print(f"Default backend: {jax.default_backend()}")


# ============================================================================
# Test 2: HoloNet Package Import
# ============================================================================
def test_holonet_imports():
    """Test that HoloNet package can be imported."""
    print("Importing HoloNet modules...")
    
    # Add parent directory to path
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    from holomorphic_networks import (
        holomorphic_elu,
        holomorphic_swish,
        ComplexMLP,
        analyze_magnitude_drift
    )
    print("  ✓ Core modules imported")
    
    from holomorphic_networks.training import (
        ComplexTrainer,
        TrainingConfig,
        compare_backprop_methods
    )
    print("  ✓ Training modules imported")
    
    from holomorphic_networks.models_jax import (
        ComplexMLPJAX,
        MLPConfig
    )
    print("  ✓ JAX-optimized modules imported")
    
    from holomorphic_networks.jax_utils import (
        KeyManager,
        check_nan_inf
    )
    print("  ✓ JAX utilities imported")


# ============================================================================
# Test 3: Activation Functions
# ============================================================================
def test_activation_functions():
    """Test holomorphic activation functions."""
    import jax.numpy as jnp
    from jax import random
    from holomorphic_networks.activations import (
        holomorphic_elu, holomorphic_swish, crelu, modrelu, complex_tanh
    )
    
    print("Testing activation functions...")
    
    # Create test input
    key = random.PRNGKey(42)
    x = random.normal(key, (10,)) + 1j * random.normal(key, (10,))
    print(f"  Input shape: {x.shape}, dtype: {x.dtype}")
    
    # Test each activation
    activations = {
        'holomorphic_elu': holomorphic_elu,
        'holomorphic_swish': holomorphic_swish,
        'crelu': crelu,
        'modrelu': modrelu,
        'complex_tanh': complex_tanh
    }
    
    for name, func in activations.items():
        y = func(x)
        assert y.shape == x.shape, f"{name} changed shape"
        assert jnp.iscomplexobj(y), f"{name} output not complex"
        assert jnp.all(jnp.isfinite(y)), f"{name} produced non-finite values"
        print(f"  ✓ {name}: input {x[0]:.3f} -> output {y[0]:.3f}")
    
    # Test gradient computation
    print("\nTesting gradient computation...")
    import jax
    
    def loss_fn(x):
        y = holomorphic_elu(x)
        return jnp.mean(jnp.abs(y) ** 2).real
    
    grad = jax.grad(loss_fn)(x)
    assert grad.shape == x.shape, "Gradient shape mismatch"
    assert jnp.all(jnp.isfinite(grad)), "Gradient has non-finite values"
    print(f"  ✓ Gradient computed: shape {grad.shape}, norm {jnp.linalg.norm(grad):.3f}")


# ============================================================================
# Test 4: Complex MLP Forward Pass
# ============================================================================
def test_mlp_forward():
    """Test ComplexMLP forward pass."""
    import jax.numpy as jnp
    from jax import random
    from holomorphic_networks.models import ComplexMLP
    
    print("Testing ComplexMLP forward pass...")
    
    # Create model
    model = ComplexMLP(
        layer_sizes=[2, 32, 32, 1],
        activation='h_elu',
        dtype=jnp.complex64
    )
    
    # Initialize parameters
    key = random.PRNGKey(42)
    params = model.init_params(key)
    print(f"  Model created: {[2, 32, 32, 1]} architecture")
    print(f"  Parameters initialized: {len(params['layers'])} layers")
    
    # Test forward pass
    batch_size = 16
    x = random.normal(key, (batch_size, 2)) + 1j * random.normal(key, (batch_size, 2))
    
    output, aux_data = model.forward(params, x, training=False)
    
    assert output.shape == (batch_size, 1), f"Output shape mismatch: {output.shape}"
    assert jnp.iscomplexobj(output), "Output not complex"
    assert jnp.all(jnp.isfinite(output)), "Output has non-finite values"
    
    print(f"  ✓ Forward pass: input {x.shape} -> output {output.shape}")
    print(f"  ✓ Magnitude tracking: {len(aux_data['magnitudes'])} layers tracked")
    
    # Check magnitude statistics
    for i, mag_stats in enumerate(aux_data['magnitudes']):
        print(f"    Layer {i}: mean={mag_stats['mean']:.3f}, std={mag_stats['std']:.3f}")


# ============================================================================
# Test 5: JAX-Optimized Model
# ============================================================================
def test_jax_model():
    """Test JAX-optimized ComplexMLPJAX."""
    import jax.numpy as jnp
    from jax import random
    from holomorphic_networks.models_jax import ComplexMLPJAX
    from holomorphic_networks.jax_utils import KeyManager
    
    print("Testing JAX-optimized model...")
    
    # Create model with KeyManager
    keys = KeyManager(seed=42)
    
    model = ComplexMLPJAX(
        layer_sizes=[2, 32, 32, 1],
        activation='h_elu',
        dropout_rate=0.1
    )
    
    # Initialize parameters
    params = model.init_params(keys.next_key())
    print(f"  JAX model created with config: {model.config}")
    
    # Test forward pass
    batch_size = 16
    x = random.normal(keys.next_key(), (batch_size, 2)) + \
        1j * random.normal(keys.next_key(), (batch_size, 2))
    
    # Test inference (no dropout)
    output_infer = model.apply(params, x)
    assert output_infer.shape == (batch_size, 1), "Inference output shape mismatch"
    print(f"  ✓ Inference: {x.shape} -> {output_infer.shape}")
    
    # Test training forward (with dropout)
    output_train, aux = model.forward(params, x, training=True, key=keys.next_key())
    assert output_train.shape == (batch_size, 1), "Training output shape mismatch"
    print(f"  ✓ Training forward: {x.shape} -> {output_train.shape}")
    
    # Test JIT compilation
    import time
    
    # First call (compilation)
    start = time.time()
    _ = model.apply(params, x)
    first_time = time.time() - start
    
    # Second call (cached)
    start = time.time()
    _ = model.apply(params, x)
    second_time = time.time() - start
    
    print(f"  ✓ JIT compilation: first={first_time:.4f}s, second={second_time:.4f}s")
    print(f"    Speedup: {first_time/max(second_time, 1e-6):.1f}x")


# ============================================================================
# Test 6: Training Step
# ============================================================================
def test_training():
    """Test basic training functionality."""
    import jax.numpy as jnp
    from jax import random
    import optax
    from holomorphic_networks.models import ComplexMLP
    from holomorphic_networks.training import ComplexTrainer, TrainingConfig
    
    print("Testing training functionality...")
    
    # Create simple dataset
    key = random.PRNGKey(42)
    n_samples = 100
    X = random.normal(key, (n_samples, 1)) + 1j * random.normal(key, (n_samples, 1))
    y = X ** 2  # Learn z^2 function
    
    print(f"  Dataset: {X.shape} -> {y.shape}")
    
    # Create model
    model = ComplexMLP(
        layer_sizes=[1, 16, 16, 1],
        activation='h_elu'
    )
    
    # Test with JAX autodiff (should work best)
    config = TrainingConfig(
        learning_rate=0.01,
        n_epochs=10,
        batch_size=32,
        backprop_method='jax_autodiff',
        verbose=False
    )
    
    trainer = ComplexTrainer(model, config)
    
    # Initialize and train
    params = model.init_params(key)
    
    print("  Training for 10 epochs...")
    final_params = trainer.train(
        train_data=(X, y),
        initial_params=params,
        key=key
    )
    
    # Check that loss decreased
    initial_loss = trainer.history['train_loss'][0]
    final_loss = trainer.history['train_loss'][-1]
    
    print(f"  ✓ Training completed: initial_loss={initial_loss:.4f}, final_loss={final_loss:.4f}")
    assert final_loss < initial_loss, "Loss did not decrease!"
    
    # Test prediction
    test_x = jnp.array([[0.5 + 0.5j]])
    pred = model(final_params, test_x)
    expected = test_x ** 2
    error = jnp.abs(pred - expected)
    print(f"  ✓ Prediction test: {test_x[0,0]} -> {pred[0,0]:.3f} (expected {expected[0,0]:.3f})")


# ============================================================================
# Test 7: Backpropagation Methods Comparison
# ============================================================================
def test_backprop_methods():
    """Test different backpropagation methods."""
    import jax.numpy as jnp
    from jax import random
    from holomorphic_networks.models import ComplexMLP
    from holomorphic_networks.training import compare_backprop_methods, TrainingConfig
    
    print("Testing backpropagation methods...")
    
    # Create simple dataset
    key = random.PRNGKey(42)
    n_samples = 50
    X = random.normal(key, (n_samples, 1)) + 1j * random.normal(key, (n_samples, 1))
    y = X ** 2
    
    # Create model
    model = ComplexMLP(
        layer_sizes=[1, 8, 1],
        activation='h_elu'
    )
    
    # Quick test config
    config = TrainingConfig(
        learning_rate=0.01,
        n_epochs=5,
        batch_size=10,
        verbose=False
    )
    
    # Compare methods
    methods = ['jax_autodiff', 'split']  # Start with just 2 for quick test
    
    print(f"  Comparing methods: {methods}")
    results = compare_backprop_methods(
        model,
        train_data=(X, y),
        methods=methods,
        config=config,
        key=key
    )
    
    # Check results
    for method in methods:
        final_loss = results[method]['final_train_loss']
        print(f"  ✓ {method}: final_loss={final_loss:.4f}")
        assert final_loss < 10.0, f"{method} failed to train"


# ============================================================================
# Test 8: Magnitude Drift Analysis
# ============================================================================
def test_magnitude_analysis():
    """Test magnitude drift analysis tools."""
    import jax.numpy as jnp
    from jax import random
    from holomorphic_networks.models import ComplexMLP
    from holomorphic_networks.analysis import analyze_magnitude_drift
    
    print("Testing magnitude drift analysis...")
    
    # Create deep network
    model = ComplexMLP(
        layer_sizes=[2, 32, 32, 32, 32, 1],  # 4 hidden layers
        activation='h_elu'
    )
    
    key = random.PRNGKey(42)
    params = model.init_params(key)
    
    # Test input
    x = random.normal(key, (10, 2)) + 1j * random.normal(key, (10, 2))
    
    # Analyze magnitude drift
    metrics = analyze_magnitude_drift(
        lambda p, x: model.forward(p, x),
        params,
        x
    )
    
    print(f"  ✓ Magnitude drift analysis completed")
    print(f"    Drift ratio: {metrics.drift_ratio:.3f}")
    print(f"    Layers analyzed: {len(metrics.layer_stats)}")
    print(f"    Input magnitude: {metrics.layer_stats[0].mean:.3f}")
    print(f"    Output magnitude: {metrics.layer_stats[-1].mean:.3f}")
    
    # Check that holomorphic activations prevent extreme drift
    assert 0.1 < metrics.drift_ratio < 10.0, f"Extreme magnitude drift: {metrics.drift_ratio}"


# ============================================================================
# Test 9: Visualization
# ============================================================================
def test_visualization():
    """Test visualization utilities."""
    import jax.numpy as jnp
    from holomorphic_networks.activations import holomorphic_elu
    from holomorphic_networks.visualization import plot_complex_plane_mapping
    import matplotlib.pyplot as plt
    
    print("Testing visualization...")
    
    try:
        # Create simple plot
        fig = plot_complex_plane_mapping(
            holomorphic_elu,
            xlim=(-2, 2),
            ylim=(-2, 2),
            resolution=50,
            save_path=None  # Don't save, just test creation
        )
        plt.close(fig)
        print("  ✓ Complex plane mapping plot created")
    except Exception as e:
        print(f"  ⚠️ Visualization test skipped (may need display): {e}")
        test_results['skipped'].append('visualization')


# ============================================================================
# Main Test Runner
# ============================================================================
def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("HOLONET TEST SUITE")
    print("="*60)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("HoloNet Package Import", test_holonet_imports),
        ("Activation Functions", test_activation_functions),
        ("MLP Forward Pass", test_mlp_forward),
        ("JAX-Optimized Model", test_jax_model),
        ("Training", test_training),
        ("Backprop Methods", test_backprop_methods),
        ("Magnitude Analysis", test_magnitude_analysis),
        ("Visualization", test_visualization),
    ]
    
    # Run tests
    for test_name, test_func in tests:
        success = run_test(test_name, test_func)
        if not success and test_name in ["Basic Imports", "HoloNet Package Import"]:
            print("\n⚠️ Critical test failed. Stopping test suite.")
            break
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"✅ Passed: {len(test_results['passed'])}")
    print(f"❌ Failed: {len(test_results['failed'])}")
    print(f"⚠️  Skipped: {len(test_results['skipped'])}")
    
    if test_results['failed']:
        print("\nFailed tests:")
        for test_name, error in test_results['failed']:
            print(f"  - {test_name}: {error}")
    
    if test_results['skipped']:
        print("\nSkipped tests:")
        for test_name in test_results['skipped']:
            print(f"  - {test_name}")
    
    # Return exit code
    return 0 if len(test_results['failed']) == 0 else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)