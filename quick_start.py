#!/usr/bin/env python3
"""
Quick start script for HoloNet - Run this after installation to see it in action!
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
import numpy as np


def main():
    print("\n" + "="*60)
    print("🚀 HOLONET QUICK START")
    print("="*60)
    
    # ========================================================================
    # 1. Test Complex Activation Functions
    # ========================================================================
    print("\n1️⃣ Testing Holomorphic Activation Functions")
    print("-" * 40)
    
    from holomorphic_networks.activations import holomorphic_elu, holomorphic_swish, crelu, modrelu
    
    # Create complex input
    z = jnp.array([1.0 + 1.0j, -1.0 + 0.5j, 0.5 - 0.8j])
    
    print(f"Input: {z}")
    print(f"H-ELU output: {holomorphic_elu(z)}")
    print(f"H-Swish output: {holomorphic_swish(z)}")
    print(f"CReLU output: {crelu(z)}")
    
    # ========================================================================
    # 2. Create and Test a Complex Neural Network
    # ========================================================================
    print("\n2️⃣ Creating Complex Neural Network")
    print("-" * 40)
    
    from holomorphic_networks.models import ComplexMLP
    
    # Create model
    model = ComplexMLP(
        layer_sizes=[2, 32, 32, 1],
        activation='h_elu',
        dtype=jnp.complex64
    )
    
    # Initialize parameters
    key = random.PRNGKey(42)
    params = model.init_params(key)
    
    # Test forward pass
    x = random.normal(key, (5, 2)) + 1j * random.normal(key, (5, 2))
    output, aux_data = model.forward(params, x)
    
    print(f"Model architecture: [2, 32, 32, 1]")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Sample output: {output[0, 0]:.4f}")
    
    # ========================================================================
    # 3. Train on Simple Complex Function (z^2)
    # ========================================================================
    print("\n3️⃣ Training to Learn z² Function")
    print("-" * 40)
    
    from holomorphic_networks.training import ComplexTrainer, TrainingConfig
    
    # Generate training data
    n_samples = 200
    key_data = random.PRNGKey(123)
    X_train = random.normal(key_data, (n_samples, 1)) + 1j * random.normal(key_data, (n_samples, 1))
    X_train = X_train * 0.5  # Scale down to avoid explosion
    y_train = X_train ** 2
    
    # Create simple model
    simple_model = ComplexMLP(
        layer_sizes=[1, 16, 16, 1],
        activation='h_elu'
    )
    
    # Training configuration - very conservative settings
    config = TrainingConfig(
        learning_rate=0.0001,  # Much lower learning rate
        n_epochs=50,  # Fewer epochs for stability testing
        batch_size=8,  # Very small batches
        backprop_method='jax_autodiff',
        verbose=True  # Enable verbose to see training progress
    )
    
    # Train
    trainer = ComplexTrainer(simple_model, config)
    initial_params = simple_model.init_params(random.PRNGKey(456))
    
    print("Training for 100 epochs...")
    final_params = trainer.train(
        train_data=(X_train, y_train),
        initial_params=initial_params
    )
    
    # Evaluate
    initial_loss = trainer.history['train_loss'][0]
    final_loss = trainer.history['train_loss'][-1]
    
    print(f"Initial loss: {initial_loss:.4f}")
    print(f"Final loss: {final_loss:.4f}")
    print(f"Improvement: {(1 - final_loss/initial_loss)*100:.1f}%")
    
    # Test prediction
    test_vals = jnp.array([[0.5 + 0.5j], [1.0 + 0.0j], [0.0 + 1.0j]])
    predictions = simple_model(final_params, test_vals)
    expected = test_vals ** 2
    
    print("\nTest predictions:")
    for i in range(len(test_vals)):
        print(f"  {test_vals[i, 0]:.2f} → Predicted: {predictions[i, 0]:.3f}, Expected: {expected[i, 0]:.3f}")
    
    # ========================================================================
    # 4. Compare Backpropagation Methods
    # ========================================================================
    print("\n4️⃣ Comparing Backpropagation Methods")
    print("-" * 40)
    
    from holomorphic_networks.training import compare_backprop_methods
    
    # Quick comparison with fewer epochs
    quick_config = TrainingConfig(
        learning_rate=0.001,  # Lower learning rate
        n_epochs=20,
        batch_size=16,  # Smaller batches
        verbose=False
    )
    
    methods = ['jax_autodiff', 'wirtinger', 'split']
    print(f"Comparing: {methods}")
    
    results = compare_backprop_methods(
        simple_model,
        train_data=(X_train[:100], y_train[:100]),  # Smaller dataset for speed
        methods=methods,
        config=quick_config
    )
    
    print("\nResults:")
    for method in methods:
        loss = results[method]['final_train_loss']
        print(f"  {method:15s}: final_loss = {loss:.6f}")
    
    # ========================================================================
    # 5. Analyze Magnitude Drift
    # ========================================================================
    print("\n5️⃣ Analyzing Magnitude Drift in Deep Networks")
    print("-" * 40)
    
    from holomorphic_networks.analysis import analyze_magnitude_drift
    
    # Create deep network
    deep_model = ComplexMLP(
        layer_sizes=[2, 32, 32, 32, 32, 32, 1],  # 5 hidden layers
        activation='h_elu'
    )
    
    deep_params = deep_model.init_params(random.PRNGKey(789))
    test_input = random.normal(key, (10, 2)) + 1j * random.normal(key, (10, 2))
    
    # Analyze with H-ELU
    metrics_helu = analyze_magnitude_drift(
        lambda p, x: deep_model.forward(p, x),
        deep_params,
        test_input
    )
    
    print(f"Deep network with H-ELU activation:")
    print(f"  Layers: {len(metrics_helu.layer_stats)}")
    print(f"  Input magnitude: {metrics_helu.layer_stats[0].mean:.3f}")
    print(f"  Output magnitude: {metrics_helu.layer_stats[-1].mean:.3f}")
    print(f"  Drift ratio: {metrics_helu.drift_ratio:.3f}")
    
    # Compare with standard activation
    standard_model = ComplexMLP(
        layer_sizes=[2, 32, 32, 32, 32, 32, 1],
        activation='tanh'  # Standard tanh
    )
    
    standard_params = standard_model.init_params(random.PRNGKey(789))
    metrics_tanh = analyze_magnitude_drift(
        lambda p, x: standard_model.forward(p, x),
        standard_params,
        test_input
    )
    
    print(f"\nDeep network with Tanh activation:")
    print(f"  Drift ratio: {metrics_tanh.drift_ratio:.3f}")
    
    print(f"\n📊 Magnitude preservation comparison:")
    print(f"  H-ELU maintains {1/metrics_helu.drift_ratio:.1%} of input magnitude")
    print(f"  Tanh maintains {1/metrics_tanh.drift_ratio:.1%} of input magnitude")
    
    # ========================================================================
    # 6. Visualize Activation Functions
    # ========================================================================
    print("\n6️⃣ Visualizing Complex Plane Mappings")
    print("-" * 40)
    
    try:
        from holomorphic_networks.visualization import plot_complex_plane_mapping
        
        # Create visualization
        fig = plot_complex_plane_mapping(
            holomorphic_elu,
            xlim=(-2, 2),
            ylim=(-2, 2),
            resolution=100,
            save_path='h_elu_mapping.png'
        )
        plt.close(fig)
        print("✓ Visualization saved to 'h_elu_mapping.png'")
    except Exception as e:
        print(f"⚠️ Visualization skipped (may need display): {e}")
    
    # ========================================================================
    # 7. JAX-Optimized Model Performance
    # ========================================================================
    print("\n7️⃣ Testing JAX-Optimized Model")
    print("-" * 40)
    
    from holomorphic_networks.models_jax import ComplexMLPJAX
    from holomorphic_networks.models_jax import KeyManager
    import time
    
    # Create JAX-optimized model
    jax_model = ComplexMLPJAX(
        layer_sizes=[10, 128, 128, 10],
        activation='h_elu'
    )
    
    keys = KeyManager(seed=42)
    jax_params = jax_model.init_params(keys.next_key())
    
    # Large batch for performance test
    large_batch = random.normal(keys.next_key(), (1000, 10)) + \
                  1j * random.normal(keys.next_key(), (1000, 10))
    
    # Time first call (includes compilation)
    start = time.time()
    _ = jax_model.apply(jax_params, large_batch)
    first_time = time.time() - start
    
    # Time second call (uses compiled version)
    start = time.time()
    _ = jax_model.apply(jax_params, large_batch)
    second_time = time.time() - start
    
    print(f"JAX JIT Compilation Performance:")
    print(f"  First call (with compilation): {first_time:.4f}s")
    print(f"  Second call (compiled): {second_time:.4f}s")
    print(f"  Speedup: {first_time/max(second_time, 1e-6):.1f}x")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*60)
    print("✅ QUICK START COMPLETE!")
    print("="*60)
    print("\nYou've successfully:")
    print("  1. Tested holomorphic activation functions")
    print("  2. Created and run a complex neural network")
    print("  3. Trained a model to learn z² function")
    print("  4. Compared different backpropagation methods")
    print("  5. Analyzed magnitude drift in deep networks")
    print("  6. Tested JAX optimization benefits")
    
    print("\n📚 Next steps:")
    print("  - Run pytest tests/ for comprehensive testing")
    print("  - Check out experiments/synthetic.py for more examples")
    print("  - Read JAX_BEST_PRACTICES.md for optimization tips")
    print("  - Try experiments/backprop_comparison.py for detailed comparisons")
    
    print("\n🎉 HoloNet is ready to use!")


if __name__ == "__main__":
    main()