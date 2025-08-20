"""Experiment comparing different backpropagation methods for complex neural networks."""

import jax
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Optional
import time

from ..models import ComplexMLP
from ..training import compare_backprop_methods, TrainingConfig
from ..experiments.synthetic import (
    generate_complex_polynomial_data,
    generate_oscillatory_data,
    create_spiral_classification_data
)
from ..visualization import PlotConfig


def run_backprop_comparison_experiment(
    task: str = 'polynomial',
    activation: str = 'h_elu',
    hidden_dims: List[int] = [64, 64],
    n_epochs: int = 500,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    n_samples: int = 1000,
    key: Optional[random.PRNGKey] = None,
    verbose: bool = True
) -> Dict:
    """Run comprehensive comparison of backpropagation methods.
    
    Args:
        task: Type of task ('polynomial', 'oscillatory', 'spiral')
        activation: Activation function to use
        hidden_dims: Hidden layer dimensions
        n_epochs: Number of training epochs
        learning_rate: Learning rate
        batch_size: Batch size
        n_samples: Number of training samples
        key: Random key
        verbose: Whether to print progress
        
    Returns:
        Dictionary with comparison results
    """
    if key is None:
        key = random.PRNGKey(42)
    
    # Generate data based on task
    key_data, key_model = random.split(key)
    
    if task == 'polynomial':
        train_inputs, train_targets = generate_complex_polynomial_data(
            key_data, n_samples=n_samples, polynomial_degree=2
        )
        test_inputs, test_targets = generate_complex_polynomial_data(
            key_data, n_samples=200, polynomial_degree=2
        )
        output_dim = 1
        
    elif task == 'oscillatory':
        train_inputs, train_targets = generate_oscillatory_data(
            key_data, n_samples=n_samples, frequency=2.0
        )
        test_inputs, test_targets = generate_oscillatory_data(
            key_data, n_samples=200, frequency=2.0
        )
        output_dim = 1
        
    elif task == 'spiral':
        train_inputs, train_targets = create_spiral_classification_data(
            key_data, n_samples=n_samples//2, n_spirals=2
        )
        test_inputs, test_targets = create_spiral_classification_data(
            key_data, n_samples=100, n_spirals=2
        )
        output_dim = 2
        
    else:
        raise ValueError(f"Unknown task: {task}")
    
    # Create model
    input_dim = train_inputs.shape[1]
    layer_sizes = [input_dim] + hidden_dims + [output_dim]
    
    model = ComplexMLP(
        layer_sizes=layer_sizes,
        activation=activation,
        dtype=jnp.complex64
    )
    
    # Training configuration
    config = TrainingConfig(
        learning_rate=learning_rate,
        batch_size=batch_size,
        n_epochs=n_epochs,
        optimizer='adam',
        verbose=verbose,
        log_interval=100
    )
    
    # Compare backpropagation methods
    methods = ['naive_real', 'split', 'jax_autodiff', 'wirtinger']
    
    print(f"\n{'='*60}")
    print(f"Comparing Backpropagation Methods on {task.capitalize()} Task")
    print(f"{'='*60}")
    print(f"Model: {layer_sizes}")
    print(f"Activation: {activation}")
    print(f"Training samples: {n_samples}")
    print(f"Epochs: {n_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"{'='*60}\n")
    
    # Run comparison
    results = compare_backprop_methods(
        model,
        (train_inputs, train_targets),
        (test_inputs, test_targets),
        methods=methods,
        config=config,
        key=key_model
    )
    
    # Add timing information
    for method in methods:
        start_time = time.time()
        key_time, subkey = random.split(key_model)
        
        # Quick timing run (fewer epochs)
        timing_config = TrainingConfig(
            learning_rate=learning_rate,
            batch_size=batch_size,
            n_epochs=10,
            optimizer='adam',
            verbose=False,
            backprop_method=method
        )
        
        from ..training import ComplexTrainer
        trainer = ComplexTrainer(model, timing_config)
        trainer.train(
            (train_inputs[:100], train_targets[:100]),
            initial_params=model.init_params(subkey),
            key=subkey
        )
        
        elapsed = time.time() - start_time
        results[method]['time_per_epoch'] = elapsed / 10
    
    # Compute relative performance
    jax_loss = results['jax_autodiff']['final_val_loss']
    for method in methods:
        results[method]['relative_loss'] = results[method]['final_val_loss'] / jax_loss
        results[method]['converged'] = results[method]['final_val_loss'] < 0.1
    
    return {
        'task': task,
        'activation': activation,
        'model_architecture': layer_sizes,
        'results': results,
        'train_data': (train_inputs, train_targets),
        'test_data': (test_inputs, test_targets)
    }


def plot_backprop_comparison(
    experiment_results: Dict,
    save_dir: Optional[str] = None,
    config: PlotConfig = PlotConfig()
) -> plt.Figure:
    """Create visualization comparing backpropagation methods.
    
    Args:
        experiment_results: Results from run_backprop_comparison_experiment
        save_dir: Directory to save plots
        config: Plot configuration
        
    Returns:
        Matplotlib figure
    """
    results = experiment_results['results']
    methods = list(results.keys())
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), dpi=config.dpi)
    axes = axes.flatten()
    
    # Plot 1: Training loss curves
    ax = axes[0]
    for method in methods:
        history = results[method]['history']
        ax.plot(history['train_loss'], label=method, linewidth=2)
    ax.set_title('Training Loss Comparison', fontsize=config.title_size)
    ax.set_xlabel('Epoch', fontsize=config.label_size)
    ax.set_ylabel('Loss', fontsize=config.label_size)
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Validation loss curves
    ax = axes[1]
    for method in methods:
        history = results[method]['history']
        if 'val_loss' in history:
            ax.plot(history['val_loss'], label=method, linewidth=2)
    ax.set_title('Validation Loss Comparison', fontsize=config.title_size)
    ax.set_xlabel('Epoch', fontsize=config.label_size)
    ax.set_ylabel('Loss', fontsize=config.label_size)
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Final losses (bar chart)
    ax = axes[2]
    final_train_losses = [results[m]['final_train_loss'] for m in methods]
    final_val_losses = [results[m]['final_val_loss'] for m in methods]
    
    x = np.arange(len(methods))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, final_train_losses, width, label='Train', alpha=0.8)
    bars2 = ax.bar(x + width/2, final_val_losses, width, label='Validation', alpha=0.8)
    
    ax.set_title('Final Loss Comparison', fontsize=config.title_size)
    ax.set_xlabel('Method', fontsize=config.label_size)
    ax.set_ylabel('Loss', fontsize=config.label_size)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Gradient norms
    ax = axes[3]
    for method in methods:
        history = results[method]['history']
        if 'grad_norms' in history:
            ax.plot(history['grad_norms'], label=method, linewidth=2)
    ax.set_title('Gradient Norm Evolution', fontsize=config.title_size)
    ax.set_xlabel('Epoch', fontsize=config.label_size)
    ax.set_ylabel('Gradient Norm', fontsize=config.label_size)
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Time per epoch (bar chart)
    ax = axes[4]
    if 'time_per_epoch' in results[methods[0]]:
        times = [results[m]['time_per_epoch'] for m in methods]
        bars = ax.bar(methods, times, alpha=0.8, color='skyblue')
        ax.set_title('Time per Epoch', fontsize=config.title_size)
        ax.set_xlabel('Method', fontsize=config.label_size)
        ax.set_ylabel('Time (seconds)', fontsize=config.label_size)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, time_val in zip(bars, times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                   f'{time_val:.3f}s', ha='center', va='bottom', fontsize=9)
    
    # Plot 6: Relative performance
    ax = axes[5]
    relative_losses = [results[m]['relative_loss'] for m in methods]
    colors = ['green' if r <= 1.1 else 'orange' if r <= 1.5 else 'red' 
              for r in relative_losses]
    bars = ax.bar(methods, relative_losses, alpha=0.8, color=colors)
    ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, 
               label='JAX AutoDiff baseline')
    ax.set_title('Relative Performance vs JAX AutoDiff', fontsize=config.title_size)
    ax.set_xlabel('Method', fontsize=config.label_size)
    ax.set_ylabel('Relative Loss', fontsize=config.label_size)
    ax.tick_params(axis='x', rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, rel_loss in zip(bars, relative_losses):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{rel_loss:.2f}x', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle(
        f"Backpropagation Methods Comparison - {experiment_results['task'].capitalize()} Task",
        fontsize=config.title_size + 2,
        y=1.02
    )
    
    plt.tight_layout()
    
    if save_dir:
        save_path = f"{save_dir}/backprop_comparison_{experiment_results['task']}.png"
        plt.savefig(save_path, format='png', dpi=config.dpi, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    return fig


def run_comprehensive_comparison(
    tasks: List[str] = ['polynomial', 'oscillatory', 'spiral'],
    activations: List[str] = ['h_elu', 'h_swish', 'tanh'],
    save_plots: bool = True,
    plot_dir: str = './plots'
) -> Dict:
    """Run comprehensive comparison across tasks and activations.
    
    Args:
        tasks: List of tasks to evaluate
        activations: List of activation functions
        save_plots: Whether to save plots
        plot_dir: Directory for plots
        
    Returns:
        Dictionary with all results
    """
    all_results = {}
    
    for task in tasks:
        task_results = {}
        
        for activation in activations:
            print(f"\n{'='*60}")
            print(f"Task: {task}, Activation: {activation}")
            print(f"{'='*60}")
            
            # Run experiment
            experiment_results = run_backprop_comparison_experiment(
                task=task,
                activation=activation,
                n_epochs=300,  # Reduced for faster comparison
                verbose=False
            )
            
            task_results[activation] = experiment_results
            
            # Create visualization
            if save_plots:
                fig = plot_backprop_comparison(
                    experiment_results,
                    save_dir=plot_dir
                )
                plt.close(fig)
        
        all_results[task] = task_results
    
    # Print summary
    print("\n" + "="*80)
    print("COMPREHENSIVE COMPARISON SUMMARY")
    print("="*80)
    
    for task in tasks:
        print(f"\n{task.upper()} TASK:")
        print("-" * 40)
        
        for activation in activations:
            results = all_results[task][activation]['results']
            print(f"\n  Activation: {activation}")
            print(f"  {'Method':<15} {'Final Loss':<12} {'Relative':<10} {'Converged':<10}")
            print("  " + "-" * 50)
            
            for method in ['naive_real', 'split', 'jax_autodiff', 'wirtinger']:
                final_loss = results[method]['final_val_loss']
                relative = results[method]['relative_loss']
                converged = "Yes" if results[method]['converged'] else "No"
                print(f"  {method:<15} {final_loss:<12.6f} {relative:<10.3f} {converged:<10}")
    
    return all_results


if __name__ == "__main__":
    # Run single experiment
    print("Running single backpropagation comparison experiment...")
    results = run_backprop_comparison_experiment(
        task='polynomial',
        activation='h_elu',
        n_epochs=500
    )
    
    # Create visualization
    fig = plot_backprop_comparison(results, save_dir='./plots')
    plt.show()
    
    # Run comprehensive comparison (optional)
    # all_results = run_comprehensive_comparison()