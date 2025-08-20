"""Synthetic complex function learning experiments."""

import jax
import jax.numpy as jnp
from jax import random, grad, jit
import optax
from typing import Dict, List, Tuple, Callable, Optional
import matplotlib.pyplot as plt
from functools import partial

from ..models import ComplexMLP
from ..activations import get_activation, ACTIVATIONS
from ..analysis import analyze_magnitude_drift, track_magnitude_evolution
from ..visualization import plot_magnitude_drift, plot_complex_plane_mapping


def generate_complex_polynomial_data(
    key: random.PRNGKey,
    n_samples: int = 1000,
    polynomial_degree: int = 2,
    noise_level: float = 0.01,
    input_range: Tuple[float, float] = (-2.0, 2.0)
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generate synthetic complex polynomial data.
    
    Args:
        key: Random key
        n_samples: Number of samples
        polynomial_degree: Degree of polynomial (2 for z^2, 3 for z^3, etc.)
        noise_level: Gaussian noise standard deviation
        input_range: Range for real and imaginary parts
        
    Returns:
        Tuple of (inputs, targets)
    """
    key_input, key_noise = random.split(key)
    
    # Generate complex inputs
    real_parts = random.uniform(
        key_input, (n_samples,), 
        minval=input_range[0], maxval=input_range[1]
    )
    imag_parts = random.uniform(
        key_input, (n_samples,), 
        minval=input_range[0], maxval=input_range[1]
    )
    
    inputs = real_parts + 1j * imag_parts
    
    # Generate polynomial targets
    targets = inputs ** polynomial_degree
    
    # Add complex noise
    if noise_level > 0:
        noise_real = random.normal(key_noise, (n_samples,)) * noise_level
        noise_imag = random.normal(key_noise, (n_samples,)) * noise_level
        noise = noise_real + 1j * noise_imag
        targets = targets + noise
    
    return inputs.reshape(-1, 1), targets.reshape(-1, 1)


def generate_oscillatory_data(
    key: random.PRNGKey,
    n_samples: int = 1000,
    frequency: float = 1.0,
    phase_shift: float = 0.0,
    damping: float = 0.0,
    noise_level: float = 0.01
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generate oscillatory complex dynamics data.
    
    Args:
        key: Random key
        n_samples: Number of samples
        frequency: Oscillation frequency
        phase_shift: Phase shift in radians
        damping: Exponential damping factor
        noise_level: Noise standard deviation
        
    Returns:
        Tuple of (inputs, targets)
    """
    key_t, key_noise = random.split(key)
    
    # Generate time points
    t = random.uniform(key_t, (n_samples,), minval=0.0, maxval=4*jnp.pi)
    
    # Generate oscillatory dynamics
    inputs = t.reshape(-1, 1) + 0j  # Time as complex input
    
    # Complex oscillation: exp(-damping*t) * exp(i*(frequency*t + phase))
    targets = (
        jnp.exp(-damping * t) * 
        jnp.exp(1j * (frequency * t + phase_shift))
    )
    
    # Add noise
    if noise_level > 0:
        noise_real = random.normal(key_noise, (n_samples,)) * noise_level
        noise_imag = random.normal(key_noise, (n_samples,)) * noise_level
        noise = noise_real + 1j * noise_imag
        targets = targets + noise
    
    return inputs, targets.reshape(-1, 1)


def create_spiral_classification_data(
    key: random.PRNGKey,
    n_samples: int = 1000,
    n_spirals: int = 2,
    noise_level: float = 0.1
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generate complex spiral classification data.
    
    Args:
        key: Random key
        n_samples: Number of samples per spiral
        n_spirals: Number of spirals
        noise_level: Noise level
        
    Returns:
        Tuple of (complex_points, labels)
    """
    key_spiral, key_noise = random.split(key)
    
    all_points = []
    all_labels = []
    
    for spiral_idx in range(n_spirals):
        # Generate spiral parameters
        t = jnp.linspace(0, 4*jnp.pi, n_samples)
        radius = t / (4*jnp.pi)
        
        # Spiral angle offset for each spiral
        angle_offset = spiral_idx * 2 * jnp.pi / n_spirals
        angles = t + angle_offset
        
        # Convert to complex coordinates
        points = radius * jnp.exp(1j * angles)
        
        # Add noise
        if noise_level > 0:
            noise_real = random.normal(key_noise, (n_samples,)) * noise_level
            noise_imag = random.normal(key_noise, (n_samples,)) * noise_level
            noise = noise_real + 1j * noise_imag
            points = points + noise
        
        all_points.append(points)
        all_labels.append(jnp.full(n_samples, spiral_idx))
    
    # Combine all spirals
    complex_points = jnp.concatenate(all_points).reshape(-1, 1)
    labels = jnp.concatenate(all_labels)
    
    # Convert labels to one-hot
    labels_onehot = jax.nn.one_hot(labels, n_spirals).astype(jnp.complex64)
    
    return complex_points, labels_onehot


@partial(jit, static_argnums=(0, 4))
def train_step(
    model: ComplexMLP,
    params: Dict,
    opt_state,
    batch: Tuple[jnp.ndarray, jnp.ndarray],
    optimizer
) -> Tuple[Dict, any, float, Dict]:
    """Single training step.
    
    Args:
        model: ComplexMLP instance
        params: Model parameters
        opt_state: Optimizer state
        batch: Batch of (inputs, targets)
        optimizer: Optax optimizer
        
    Returns:
        Tuple of (updated_params, updated_opt_state, loss, metrics)
    """
    inputs, targets = batch
    
    def loss_fn(p):
        predictions, aux_data = model.forward(p, inputs, training=True)
        mse_loss = jnp.mean(jnp.abs(predictions - targets) ** 2)
        
        # Add magnitude drift regularization
        magnitude_penalty = 0.0
        if 'magnitudes' in aux_data:
            for mag_dict in aux_data['magnitudes']:
                # Penalize very large magnitudes
                magnitude_penalty += jnp.maximum(0.0, mag_dict['max'] - 10.0) ** 2
        
        total_loss = mse_loss + 0.001 * magnitude_penalty
        return total_loss, (mse_loss, aux_data)
    
    (loss, (mse_loss, aux_data)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    
    # Compute metrics
    predictions, _ = model.forward(params, inputs, training=False)
    accuracy = jnp.mean(jnp.abs(predictions - targets) ** 2)
    
    metrics = {
        'mse_loss': float(mse_loss),
        'total_loss': float(loss),
        'accuracy': float(accuracy),
        'magnitude_stats': aux_data.get('magnitudes', [])
    }
    
    return params, opt_state, loss, metrics


def run_polynomial_experiment(
    activation_name: str = 'h_elu',
    polynomial_degree: int = 2,
    n_epochs: int = 1000,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    hidden_dims: List[int] = [64, 64],
    key: Optional[random.PRNGKey] = None
) -> Dict:
    """Run polynomial learning experiment.
    
    Args:
        activation_name: Name of activation function
        polynomial_degree: Degree of polynomial to learn
        n_epochs: Number of training epochs
        learning_rate: Learning rate
        batch_size: Batch size
        hidden_dims: Hidden layer dimensions
        key: Random key (if None, uses default)
        
    Returns:
        Dictionary with experimental results
    """
    if key is None:
        key = random.PRNGKey(42)
    
    # Generate data
    key_data, key_model, key_train = random.split(key, 3)
    
    train_inputs, train_targets = generate_complex_polynomial_data(
        key_data, n_samples=1000, polynomial_degree=polynomial_degree
    )
    
    test_inputs, test_targets = generate_complex_polynomial_data(
        key_data, n_samples=200, polynomial_degree=polynomial_degree
    )
    
    # Create model
    layer_sizes = [1] + hidden_dims + [1]
    model = ComplexMLP(
        layer_sizes=layer_sizes,
        activation=activation_name,
        dtype=jnp.complex64
    )
    
    params = model.init_params(key_model)
    
    # Setup optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    
    # Training loop
    train_losses = []
    test_losses = []
    magnitude_metrics = []
    
    n_batches = len(train_inputs) // batch_size
    
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        
        # Shuffle data
        perm = random.permutation(key_train, len(train_inputs))
        train_inputs_shuffled = train_inputs[perm]
        train_targets_shuffled = train_targets[perm]
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            
            batch_inputs = train_inputs_shuffled[start_idx:end_idx]
            batch_targets = train_targets_shuffled[start_idx:end_idx]
            
            params, opt_state, loss, metrics = train_step(
                model, params, opt_state, (batch_inputs, batch_targets), optimizer
            )
            
            epoch_loss += loss
        
        # Record metrics
        avg_epoch_loss = epoch_loss / n_batches
        train_losses.append(float(avg_epoch_loss))
        
        # Test loss
        test_predictions, test_aux = model.forward(params, test_inputs, training=False)
        test_loss = jnp.mean(jnp.abs(test_predictions - test_targets) ** 2)
        test_losses.append(float(test_loss))
        
        # Magnitude drift analysis
        if epoch % 100 == 0:
            drift_metrics = analyze_magnitude_drift(
                lambda p, x: model.forward(p, x, training=False),
                params,
                test_inputs[:10]  # Subset for efficiency
            )
            magnitude_metrics.append(drift_metrics.to_dict())
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch}: Train Loss = {avg_epoch_loss:.6f}, Test Loss = {test_loss:.6f}")
    
    # Final evaluation
    final_predictions, final_aux = model.forward(params, test_inputs, training=False)
    final_test_loss = jnp.mean(jnp.abs(final_predictions - test_targets) ** 2)
    
    # Comprehensive magnitude analysis
    final_magnitude_evolution = track_magnitude_evolution(
        lambda p, x: model.forward(p, x, training=False),
        params,
        test_inputs
    )
    
    return {
        'activation': activation_name,
        'polynomial_degree': polynomial_degree,
        'final_test_loss': float(final_test_loss),
        'train_losses': train_losses,
        'test_losses': test_losses,
        'magnitude_metrics': magnitude_metrics,
        'final_magnitude_evolution': final_magnitude_evolution,
        'model_params': params,
        'test_inputs': test_inputs,
        'test_targets': test_targets,
        'final_predictions': final_predictions
    }


def compare_activations_polynomial(
    polynomial_degree: int = 2,
    activations: List[str] = ['h_elu', 'h_swish', 'crelu', 'modrelu', 'tanh'],
    n_epochs: int = 1000,
    save_plots: bool = True,
    plot_dir: str = './plots'
) -> Dict:
    """Compare different activation functions on polynomial learning.
    
    Args:
        polynomial_degree: Degree of polynomial to learn
        activations: List of activation function names
        n_epochs: Number of training epochs
        save_plots: Whether to save visualization plots
        plot_dir: Directory to save plots
        
    Returns:
        Dictionary with comparison results
    """
    results = {}
    
    for activation in activations:
        print(f"\nTraining with {activation} activation...")
        
        result = run_polynomial_experiment(
            activation_name=activation,
            polynomial_degree=polynomial_degree,
            n_epochs=n_epochs
        )
        
        results[activation] = result
        
        if save_plots:
            # Plot magnitude drift
            fig = plot_magnitude_drift(
                result['final_magnitude_evolution'],
                save_path=f"{plot_dir}/magnitude_drift_{activation}_poly{polynomial_degree}.png"
            )
            plt.close(fig)
            
            # Plot activation function mapping
            activation_fn = get_activation(activation)
            fig = plot_complex_plane_mapping(
                activation_fn,
                save_path=f"{plot_dir}/activation_mapping_{activation}.png"
            )
            plt.close(fig)
    
    # Create comparison summary
    summary = {}
    for activation, result in results.items():
        summary[activation] = {
            'final_loss': result['final_test_loss'],
            'drift_ratio': result['final_magnitude_evolution']['mean_drift_ratio'],
            'magnitude_variance': result['final_magnitude_evolution']['std_drift_ratio']
        }
    
    print("\n" + "="*50)
    print("COMPARISON SUMMARY")
    print("="*50)
    print(f"{'Activation':<12} {'Final Loss':<12} {'Drift Ratio':<12} {'Mag Variance':<12}")
    print("-" * 50)
    
    for activation, metrics in summary.items():
        print(f"{activation:<12} {metrics['final_loss']:<12.6f} "
              f"{metrics['drift_ratio']:<12.3f} {metrics['magnitude_variance']:<12.3f}")
    
    return {
        'detailed_results': results,
        'summary': summary,
        'polynomial_degree': polynomial_degree
    }


if __name__ == "__main__":
    # Run comparison experiment
    results = compare_activations_polynomial(
        polynomial_degree=2,
        activations=['h_elu', 'h_swish', 'crelu', 'tanh'],
        n_epochs=500
    )