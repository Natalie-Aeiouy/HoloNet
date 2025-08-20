"""Visualization utilities for complex neural networks."""

from typing import Dict, List, Optional, Tuple, Any, Union
import jax.numpy as jnp
from jax import Array
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import hsv_to_rgb
from matplotlib.animation import FuncAnimation
import seaborn as sns
from dataclasses import dataclass


# Set matplotlib style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


@dataclass
class PlotConfig:
    """Configuration for plot styling."""
    figsize: Tuple[int, int] = (12, 8)
    dpi: int = 300
    save_format: str = 'png'
    color_palette: str = 'viridis'
    font_size: int = 12
    title_size: int = 14
    label_size: int = 10


def complex_to_color(z: Array, colormap: str = 'hsv') -> Array:
    """Convert complex numbers to colors using phase and magnitude.
    
    Args:
        z: Complex array
        colormap: Color mapping scheme ('hsv', 'magnitude', 'phase')
        
    Returns:
        RGB color array
    """
    if colormap == 'hsv':
        # Use phase for hue, magnitude for saturation/value
        phases = jnp.angle(z) / (2 * jnp.pi) + 0.5  # Map to [0, 1]
        magnitudes = jnp.abs(z)
        
        # Normalize magnitudes for better visualization
        mag_max = jnp.max(magnitudes)
        if mag_max > 0:
            magnitudes = magnitudes / mag_max
        
        # Create HSV colors
        h = phases
        s = jnp.ones_like(magnitudes)  # Full saturation
        v = magnitudes  # Value represents magnitude
        
        # Convert to RGB
        hsv = jnp.stack([h, s, v], axis=-1)
        rgb = np.array([hsv_to_rgb(pixel) for pixel in hsv.reshape(-1, 3)])
        return rgb.reshape(hsv.shape)
    
    elif colormap == 'magnitude':
        # Color by magnitude only
        magnitudes = jnp.abs(z)
        mag_max = jnp.max(magnitudes)
        if mag_max > 0:
            magnitudes = magnitudes / mag_max
        return plt.cm.viridis(magnitudes)[..., :3]
    
    elif colormap == 'phase':
        # Color by phase only
        phases = (jnp.angle(z) / (2 * jnp.pi) + 0.5) % 1.0
        return plt.cm.hsv(phases)[..., :3]


def plot_complex_plane_mapping(
    activation_fn: callable,
    xlim: Tuple[float, float] = (-3, 3),
    ylim: Tuple[float, float] = (-3, 3),
    resolution: int = 200,
    config: PlotConfig = PlotConfig(),
    save_path: Optional[str] = None
) -> plt.Figure:
    """Visualize how an activation function maps the complex plane.
    
    Args:
        activation_fn: Complex activation function
        xlim: Real axis limits
        ylim: Imaginary axis limits
        resolution: Grid resolution
        config: Plot configuration
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    # Create complex grid
    x = jnp.linspace(xlim[0], xlim[1], resolution)
    y = jnp.linspace(ylim[0], ylim[1], resolution)
    X, Y = jnp.meshgrid(x, y)
    Z = X + 1j * Y
    
    # Apply activation function
    W = activation_fn(Z)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=config.dpi)
    
    # Input plane (colored by phase)
    input_colors = complex_to_color(Z, 'phase')
    axes[0].imshow(input_colors, extent=[xlim[0], xlim[1], ylim[0], ylim[1]], 
                   origin='lower', aspect='equal')
    axes[0].set_title('Input Complex Plane', fontsize=config.title_size)
    axes[0].set_xlabel('Real Part', fontsize=config.label_size)
    axes[0].set_ylabel('Imaginary Part', fontsize=config.label_size)
    axes[0].grid(True, alpha=0.3)
    
    # Output plane (colored by phase)
    output_colors = complex_to_color(W, 'phase')
    axes[1].imshow(output_colors, extent=[jnp.real(W).min(), jnp.real(W).max(),
                                         jnp.imag(W).min(), jnp.imag(W).max()],
                   origin='lower', aspect='equal')
    axes[1].set_title('Output Complex Plane', fontsize=config.title_size)
    axes[1].set_xlabel('Real Part', fontsize=config.label_size)
    axes[1].set_ylabel('Imaginary Part', fontsize=config.label_size)
    axes[1].grid(True, alpha=0.3)
    
    # Magnitude mapping
    magnitude_plot = axes[2].imshow(jnp.abs(W), extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
                                    origin='lower', aspect='equal', cmap=config.color_palette)
    axes[2].set_title('Output Magnitude', fontsize=config.title_size)
    axes[2].set_xlabel('Real Part', fontsize=config.label_size)
    axes[2].set_ylabel('Imaginary Part', fontsize=config.label_size)
    plt.colorbar(magnitude_plot, ax=axes[2])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, format=config.save_format, dpi=config.dpi, bbox_inches='tight')
    
    return fig


def plot_magnitude_drift(
    magnitude_evolution: Dict[str, Array],
    activation_names: Optional[List[str]] = None,
    config: PlotConfig = PlotConfig(),
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot magnitude drift through network layers.
    
    Args:
        magnitude_evolution: Dictionary with magnitude statistics
        activation_names: Names of activation functions being compared
        config: Plot configuration
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=config.figsize, dpi=config.dpi)
    axes = axes.flatten()
    
    layer_indices = jnp.arange(len(magnitude_evolution['layer_means']))
    
    # Plot 1: Mean magnitude evolution
    axes[0].plot(layer_indices, magnitude_evolution['layer_means'], 'o-', linewidth=2, markersize=6)
    axes[0].fill_between(layer_indices, 
                        magnitude_evolution['layer_means'] - magnitude_evolution['layer_stds'],
                        magnitude_evolution['layer_means'] + magnitude_evolution['layer_stds'],
                        alpha=0.3)
    axes[0].set_title('Mean Magnitude Evolution', fontsize=config.title_size)
    axes[0].set_xlabel('Layer Index', fontsize=config.label_size)
    axes[0].set_ylabel('Mean Magnitude', fontsize=config.label_size)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')
    
    # Plot 2: Magnitude range (min to max)
    axes[1].fill_between(layer_indices, magnitude_evolution['layer_mins'], 
                        magnitude_evolution['layer_maxs'], alpha=0.6, label='Min-Max Range')
    axes[1].plot(layer_indices, magnitude_evolution['layer_means'], 'k-', linewidth=2, label='Mean')
    axes[1].set_title('Magnitude Range per Layer', fontsize=config.title_size)
    axes[1].set_xlabel('Layer Index', fontsize=config.label_size)
    axes[1].set_ylabel('Magnitude', fontsize=config.label_size)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')
    
    # Plot 3: Drift ratio distribution
    if 'drift_ratios' in magnitude_evolution:
        axes[2].hist(magnitude_evolution['drift_ratios'], bins=30, alpha=0.7, density=True)
        axes[2].axvline(magnitude_evolution['mean_drift_ratio'], color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {magnitude_evolution["mean_drift_ratio"]:.3f}')
        axes[2].set_title('Drift Ratio Distribution', fontsize=config.title_size)
        axes[2].set_xlabel('Output/Input Magnitude Ratio', fontsize=config.label_size)
        axes[2].set_ylabel('Density', fontsize=config.label_size)
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    
    # Plot 4: Cumulative change distribution
    if 'cumulative_changes' in magnitude_evolution:
        axes[3].hist(magnitude_evolution['cumulative_changes'], bins=30, alpha=0.7, density=True)
        axes[3].axvline(magnitude_evolution['mean_cumulative_change'], color='red', linestyle='--',
                       linewidth=2, label=f'Mean: {magnitude_evolution["mean_cumulative_change"]:.3f}')
        axes[3].set_title('Cumulative Change Distribution', fontsize=config.title_size)
        axes[3].set_xlabel('Cumulative Magnitude Change', fontsize=config.label_size)
        axes[3].set_ylabel('Density', fontsize=config.label_size)
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, format=config.save_format, dpi=config.dpi, bbox_inches='tight')
    
    return fig


def plot_activation_comparison(
    activation_results: Dict[str, Dict],
    metric: str = 'drift_ratio',
    config: PlotConfig = PlotConfig(),
    save_path: Optional[str] = None
) -> plt.Figure:
    """Compare different activation functions on a specific metric.
    
    Args:
        activation_results: Dictionary mapping activation names to results
        metric: Metric to compare ('drift_ratio', 'holomorphicity_violation', etc.)
        config: Plot configuration
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    activation_names = list(activation_results.keys())
    metric_values = []
    
    for name in activation_names:
        if metric in activation_results[name]:
            if isinstance(activation_results[name][metric], dict):
                # Handle nested metrics
                metric_values.append(activation_results[name][metric]['mean'])
            else:
                metric_values.append(activation_results[name][metric])
        else:
            metric_values.append(0)  # Default value if metric not found
    
    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
    
    bars = ax.bar(activation_names, metric_values, alpha=0.7)
    
    # Color bars based on values
    colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    ax.set_title(f'Activation Function Comparison: {metric.replace("_", " ").title()}', 
                fontsize=config.title_size)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=config.label_size)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max(metric_values),
                f'{value:.3f}', ha='center', va='bottom', fontsize=config.label_size)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, format=config.save_format, dpi=config.dpi, bbox_inches='tight')
    
    return fig


def plot_phase_evolution(
    activations: List[Array],
    layer_names: Optional[List[str]] = None,
    config: PlotConfig = PlotConfig(),
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot phase evolution through network layers.
    
    Args:
        activations: List of complex activation arrays for each layer
        layer_names: Names of layers
        config: Plot configuration
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    num_layers = len(activations)
    cols = min(4, num_layers)
    rows = (num_layers + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3), dpi=config.dpi)
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for i, activation in enumerate(activations):
        if i >= len(axes):
            break
            
        phases = jnp.angle(activation.flatten())
        
        # Create circular histogram
        ax = axes[i]
        n_bins = 30
        bins = jnp.linspace(-jnp.pi, jnp.pi, n_bins + 1)
        hist, _ = jnp.histogram(phases, bins)
        
        # Convert to polar plot
        theta = (bins[:-1] + bins[1:]) / 2
        ax.bar(theta, hist, width=2*jnp.pi/n_bins, alpha=0.7)
        
        layer_name = layer_names[i] if layer_names else f'Layer {i}'
        ax.set_title(f'{layer_name} Phase Distribution', fontsize=config.label_size)
        ax.set_theta_zero_location('E')
        ax.set_theta_direction(1)
    
    # Hide unused subplots
    for i in range(num_layers, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, format=config.save_format, dpi=config.dpi, bbox_inches='tight')
    
    return fig


def create_training_dashboard(
    training_metrics: Dict[str, List],
    magnitude_metrics: Dict[str, List],
    config: PlotConfig = PlotConfig(),
    save_path: Optional[str] = None
) -> plt.Figure:
    """Create comprehensive training dashboard.
    
    Args:
        training_metrics: Dictionary with training loss, accuracy, etc.
        magnitude_metrics: Dictionary with magnitude drift metrics over time
        config: Plot configuration
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=(16, 12), dpi=config.dpi)
    
    # Create grid layout
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Training loss
    ax1 = fig.add_subplot(gs[0, 0])
    if 'train_loss' in training_metrics:
        ax1.plot(training_metrics['train_loss'], label='Train', linewidth=2)
    if 'val_loss' in training_metrics:
        ax1.plot(training_metrics['val_loss'], label='Validation', linewidth=2)
    ax1.set_title('Training Loss', fontsize=config.title_size)
    ax1.set_xlabel('Epoch', fontsize=config.label_size)
    ax1.set_ylabel('Loss', fontsize=config.label_size)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Training accuracy (if available)
    ax2 = fig.add_subplot(gs[0, 1])
    if 'train_acc' in training_metrics:
        ax2.plot(training_metrics['train_acc'], label='Train', linewidth=2)
    if 'val_acc' in training_metrics:
        ax2.plot(training_metrics['val_acc'], label='Validation', linewidth=2)
    ax2.set_title('Accuracy', fontsize=config.title_size)
    ax2.set_xlabel('Epoch', fontsize=config.label_size)
    ax2.set_ylabel('Accuracy', fontsize=config.label_size)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Learning rate (if available)
    ax3 = fig.add_subplot(gs[0, 2])
    if 'learning_rate' in training_metrics:
        ax3.plot(training_metrics['learning_rate'], linewidth=2)
        ax3.set_title('Learning Rate', fontsize=config.title_size)
        ax3.set_xlabel('Epoch', fontsize=config.label_size)
        ax3.set_ylabel('Learning Rate', fontsize=config.label_size)
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')
    
    # Magnitude drift evolution
    ax4 = fig.add_subplot(gs[1, :])
    if 'drift_ratios' in magnitude_metrics:
        epochs = jnp.arange(len(magnitude_metrics['drift_ratios']))
        ax4.plot(epochs, magnitude_metrics['drift_ratios'], linewidth=2, color='red')
        ax4.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='No Drift')
        ax4.set_title('Magnitude Drift Evolution', fontsize=config.title_size)
        ax4.set_xlabel('Epoch', fontsize=config.label_size)
        ax4.set_ylabel('Drift Ratio', fontsize=config.label_size)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # Gradient norms
    ax5 = fig.add_subplot(gs[2, 0])
    if 'grad_norms' in training_metrics:
        ax5.plot(training_metrics['grad_norms'], linewidth=2)
        ax5.set_title('Gradient Norms', fontsize=config.title_size)
        ax5.set_xlabel('Epoch', fontsize=config.label_size)
        ax5.set_ylabel('Gradient Norm', fontsize=config.label_size)
        ax5.grid(True, alpha=0.3)
        ax5.set_yscale('log')
    
    # Parameter norms
    ax6 = fig.add_subplot(gs[2, 1])
    if 'param_norms' in training_metrics:
        ax6.plot(training_metrics['param_norms'], linewidth=2)
        ax6.set_title('Parameter Norms', fontsize=config.title_size)
        ax6.set_xlabel('Epoch', fontsize=config.label_size)
        ax6.set_ylabel('Parameter Norm', fontsize=config.label_size)
        ax6.grid(True, alpha=0.3)
    
    # Holomorphicity violations (if available)
    ax7 = fig.add_subplot(gs[2, 2])
    if 'holomorphicity_violations' in magnitude_metrics:
        violations = magnitude_metrics['holomorphicity_violations']
        if isinstance(violations[0], dict):
            # Multiple activation functions
            for act_name in violations[0].keys():
                violation_series = [v[act_name] for v in violations]
                ax7.plot(violation_series, label=act_name, linewidth=2)
            ax7.legend()
        else:
            ax7.plot(violations, linewidth=2)
        ax7.set_title('Holomorphicity Violations', fontsize=config.title_size)
        ax7.set_xlabel('Epoch', fontsize=config.label_size)
        ax7.set_ylabel('Violation', fontsize=config.label_size)
        ax7.grid(True, alpha=0.3)
        ax7.set_yscale('log')
    
    if save_path:
        plt.savefig(save_path, format=config.save_format, dpi=config.dpi, bbox_inches='tight')
    
    return fig


def animate_network_dynamics(
    model_fn: callable,
    params_history: List[Dict],
    x_test: Array,
    config: PlotConfig = PlotConfig(),
    save_path: Optional[str] = None
) -> FuncAnimation:
    """Create animation of network dynamics during training.
    
    Args:
        model_fn: Model function
        params_history: List of parameter states during training
        x_test: Test input for visualization
        config: Plot configuration
        save_path: Path to save animation
        
    Returns:
        FuncAnimation object
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=config.dpi)
    
    def animate(frame):
        # Clear axes
        for ax in axes:
            ax.clear()
        
        params = params_history[frame]
        output, aux_data = model_fn(params, x_test)
        
        # Plot input
        input_phases = jnp.angle(x_test.flatten())
        axes[0].scatter(x_test.real.flatten(), x_test.imag.flatten(), 
                       c=input_phases, cmap='hsv', alpha=0.7)
        axes[0].set_title(f'Input (Frame {frame})')
        axes[0].set_xlabel('Real')
        axes[0].set_ylabel('Imaginary')
        axes[0].grid(True, alpha=0.3)
        
        # Plot output
        output_phases = jnp.angle(output.flatten())
        axes[1].scatter(output.real.flatten(), output.imag.flatten(),
                       c=output_phases, cmap='hsv', alpha=0.7)
        axes[1].set_title(f'Output (Frame {frame})')
        axes[1].set_xlabel('Real')
        axes[1].set_ylabel('Imaginary')
        axes[1].grid(True, alpha=0.3)
        
        # Plot magnitude evolution
        if 'magnitudes' in aux_data:
            layer_mags = [mag_dict['mean'] for mag_dict in aux_data['magnitudes']]
            axes[2].plot(range(len(layer_mags)), layer_mags, 'o-', linewidth=2)
            axes[2].set_title(f'Magnitude Evolution (Frame {frame})')
            axes[2].set_xlabel('Layer')
            axes[2].set_ylabel('Mean Magnitude')
            axes[2].set_yscale('log')
            axes[2].grid(True, alpha=0.3)
    
    anim = FuncAnimation(fig, animate, frames=len(params_history), interval=200, blit=False)
    
    if save_path:
        anim.save(save_path, writer='pillow', fps=5)
    
    return anim