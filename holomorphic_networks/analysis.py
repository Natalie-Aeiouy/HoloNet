"""Magnitude drift analysis and metrics for complex neural networks."""

from typing import Dict, List, Tuple, Optional, Any
import jax
import jax.numpy as jnp
from jax import Array
import numpy as np
from dataclasses import dataclass
from functools import partial


@dataclass
class MagnitudeStats:
    """Statistics for magnitude analysis."""
    mean: float
    std: float
    max: float
    min: float
    median: float
    q25: float
    q75: float
    layer_idx: int
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for logging."""
        return {
            'mean': float(self.mean),
            'std': float(self.std),
            'max': float(self.max),
            'min': float(self.min),
            'median': float(self.median),
            'q25': float(self.q25),
            'q75': float(self.q75),
            'layer_idx': self.layer_idx
        }


@dataclass
class MagnitudeDriftMetrics:
    """Comprehensive magnitude drift metrics."""
    layer_stats: List[MagnitudeStats]
    drift_ratio: float  # max_magnitude / input_magnitude
    cumulative_change: float  # Product of layer-wise magnitude changes
    max_gradient_magnitude: float
    mean_gradient_magnitude: float
    magnitude_variance: float  # Variance across layers
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            'drift_ratio': float(self.drift_ratio),
            'cumulative_change': float(self.cumulative_change),
            'max_gradient_magnitude': float(self.max_gradient_magnitude),
            'mean_gradient_magnitude': float(self.mean_gradient_magnitude),
            'magnitude_variance': float(self.magnitude_variance),
            'layer_stats': [stats.to_dict() for stats in self.layer_stats]
        }


def compute_magnitude_stats(x: Array, layer_idx: int = 0) -> MagnitudeStats:
    """Compute comprehensive magnitude statistics for a tensor.
    
    Args:
        x: Complex tensor
        layer_idx: Layer index for tracking
        
    Returns:
        MagnitudeStats object with all statistics
    """
    magnitudes = jnp.abs(x)
    
    return MagnitudeStats(
        mean=float(jnp.mean(magnitudes)),
        std=float(jnp.std(magnitudes)),
        max=float(jnp.max(magnitudes)),
        min=float(jnp.min(magnitudes)),
        median=float(jnp.median(magnitudes)),
        q25=float(jnp.percentile(magnitudes, 25)),
        q75=float(jnp.percentile(magnitudes, 75)),
        layer_idx=layer_idx
    )


def analyze_magnitude_drift(
    model_fn: callable,
    params: Dict,
    x: Array,
    return_intermediates: bool = True
) -> MagnitudeDriftMetrics:
    """Analyze magnitude drift through the network.
    
    Args:
        model_fn: Model function that returns (output, aux_data)
        params: Model parameters
        x: Input data
        return_intermediates: Whether to return intermediate activations
        
    Returns:
        MagnitudeDriftMetrics with comprehensive analysis
    """
    # Forward pass with intermediate activations
    output, aux_data = model_fn(params, x)
    
    # Extract magnitude statistics from aux_data
    if 'magnitudes' in aux_data:
        # Use pre-computed statistics
        layer_stats = []
        for i, mag_dict in enumerate(aux_data['magnitudes']):
            # Convert JAX arrays to Python floats
            stats = MagnitudeStats(
                mean=float(mag_dict['mean']),
                std=float(mag_dict['std']),
                max=float(mag_dict['max']),
                min=float(mag_dict['min']),
                median=float(mag_dict.get('median', mag_dict['mean'])),
                q25=float(mag_dict.get('q25', mag_dict['mean'] - mag_dict['std'])),
                q75=float(mag_dict.get('q75', mag_dict['mean'] + mag_dict['std'])),
                layer_idx=i
            )
            layer_stats.append(stats)
    else:
        # Compute from activations if available
        if 'activations' in aux_data:
            layer_stats = []
            for i, activation in enumerate(aux_data['activations']):
                stats = compute_magnitude_stats(activation, i)
                layer_stats.append(stats)
        else:
            # Fallback: only input and output
            input_stats = compute_magnitude_stats(x, 0)
            output_stats = compute_magnitude_stats(output, 1)
            layer_stats = [input_stats, output_stats]
    
    # Compute drift metrics
    input_magnitude = layer_stats[0].mean
    output_magnitude = layer_stats[-1].mean
    
    drift_ratio = output_magnitude / (input_magnitude + 1e-8)
    
    # Cumulative magnitude change
    magnitude_ratios = []
    for i in range(1, len(layer_stats)):
        prev_mag = layer_stats[i-1].mean
        curr_mag = layer_stats[i].mean
        ratio = curr_mag / (prev_mag + 1e-8)
        magnitude_ratios.append(ratio)
    
    cumulative_change = float(jnp.prod(jnp.array(magnitude_ratios))) if magnitude_ratios else 1.0
    
    # Gradient magnitudes (approximate using finite differences)
    def loss_fn(p):
        out, _ = model_fn(p, x)
        return jnp.mean(jnp.abs(out) ** 2)
    
    grads = jax.grad(loss_fn)(params)
    
    # Flatten all gradients
    def extract_arrays(nested_dict):
        arrays = []
        if isinstance(nested_dict, dict):
            for v in nested_dict.values():
                arrays.extend(extract_arrays(v))
        elif isinstance(nested_dict, (list, tuple)):
            for item in nested_dict:
                arrays.extend(extract_arrays(item))
        elif hasattr(nested_dict, 'shape'):  # JAX array
            arrays.append(nested_dict)
        return arrays
    
    grad_arrays = extract_arrays(grads)
    grad_magnitudes = [jnp.abs(arr) for arr in grad_arrays if arr.size > 0]
    
    if grad_magnitudes:
        all_grad_mags = jnp.concatenate([arr.flatten() for arr in grad_magnitudes])
        max_grad_mag = float(jnp.max(all_grad_mags))
        mean_grad_mag = float(jnp.mean(all_grad_mags))
    else:
        max_grad_mag = 0.0
        mean_grad_mag = 0.0
    
    # Magnitude variance across layers
    layer_means = [stats.mean for stats in layer_stats]
    magnitude_variance = float(jnp.var(jnp.array(layer_means)))
    
    return MagnitudeDriftMetrics(
        layer_stats=layer_stats,
        drift_ratio=drift_ratio,
        cumulative_change=cumulative_change,
        max_gradient_magnitude=max_grad_mag,
        mean_gradient_magnitude=mean_grad_mag,
        magnitude_variance=magnitude_variance
    )


def track_magnitude_evolution(
    model_fn: callable,
    params: Dict,
    x_batch: Array,
    num_samples: int = 100
) -> Dict[str, Array]:
    """Track magnitude evolution over a batch of inputs.
    
    Args:
        model_fn: Model function
        params: Model parameters
        x_batch: Batch of input data
        num_samples: Number of samples to analyze
        
    Returns:
        Dictionary with magnitude evolution statistics
    """
    batch_size = min(num_samples, x_batch.shape[0])
    x_sample = x_batch[:batch_size]
    
    # Analyze each sample
    all_metrics = []
    for i in range(batch_size):
        x_single = x_sample[i:i+1]  # Keep batch dimension
        metrics = analyze_magnitude_drift(model_fn, params, x_single)
        all_metrics.append(metrics)
    
    # Aggregate statistics
    num_layers = len(all_metrics[0].layer_stats)
    
    layer_means = jnp.zeros(num_layers)
    layer_stds = jnp.zeros(num_layers)
    layer_maxs = jnp.zeros(num_layers)
    layer_mins = jnp.zeros(num_layers)
    
    for layer_idx in range(num_layers):
        layer_magnitudes = [
            metrics.layer_stats[layer_idx].mean 
            for metrics in all_metrics
        ]
        layer_means = layer_means.at[layer_idx].set(jnp.mean(jnp.array(layer_magnitudes)))
        layer_stds = layer_stds.at[layer_idx].set(jnp.std(jnp.array(layer_magnitudes)))
        layer_maxs = layer_maxs.at[layer_idx].set(jnp.max(jnp.array(layer_magnitudes)))
        layer_mins = layer_mins.at[layer_idx].set(jnp.min(jnp.array(layer_magnitudes)))
    
    # Aggregate drift metrics
    drift_ratios = [metrics.drift_ratio for metrics in all_metrics]
    cumulative_changes = [metrics.cumulative_change for metrics in all_metrics]
    
    return {
        'layer_means': layer_means,
        'layer_stds': layer_stds,
        'layer_maxs': layer_maxs,
        'layer_mins': layer_mins,
        'drift_ratios': jnp.array(drift_ratios),
        'cumulative_changes': jnp.array(cumulative_changes),
        'mean_drift_ratio': jnp.mean(jnp.array(drift_ratios)),
        'std_drift_ratio': jnp.std(jnp.array(drift_ratios)),
        'mean_cumulative_change': jnp.mean(jnp.array(cumulative_changes)),
        'std_cumulative_change': jnp.std(jnp.array(cumulative_changes))
    }


def compute_holomorphicity_violation(
    activation_fn: callable,
    z: Array,
    eps: float = 1e-6
) -> float:
    """Compute violation of Cauchy-Riemann equations for activation function.
    
    Args:
        activation_fn: Complex activation function
        z: Complex input points
        eps: Small perturbation for finite differences
        
    Returns:
        Mean violation of Cauchy-Riemann equations
    """
    def real_part(z_val):
        return activation_fn(z_val).real
    
    def imag_part(z_val):
        return activation_fn(z_val).imag
    
    # Compute partial derivatives using finite differences
    # ∂u/∂x
    du_dx = (real_part(z + eps) - real_part(z - eps)) / (2 * eps)
    
    # ∂u/∂y  
    du_dy = (real_part(z + 1j * eps) - real_part(z - 1j * eps)) / (2 * eps)
    
    # ∂v/∂x
    dv_dx = (imag_part(z + eps) - imag_part(z - eps)) / (2 * eps)
    
    # ∂v/∂y
    dv_dy = (imag_part(z + 1j * eps) - imag_part(z - 1j * eps)) / (2 * eps)
    
    # Cauchy-Riemann equations: ∂u/∂x = ∂v/∂y and ∂u/∂y = -∂v/∂x
    cr_violation_1 = jnp.abs(du_dx - dv_dy)
    cr_violation_2 = jnp.abs(du_dy + dv_dx)
    
    total_violation = cr_violation_1 + cr_violation_2
    
    return float(jnp.mean(total_violation))


def phase_coherence_analysis(x: Array) -> Dict[str, float]:
    """Analyze phase coherence in complex activations.
    
    Args:
        x: Complex tensor
        
    Returns:
        Dictionary with phase coherence metrics
    """
    phases = jnp.angle(x)
    
    # Phase variance
    phase_var = float(jnp.var(phases))
    
    # Circular variance (measure of phase dispersion)
    mean_phase_vector = jnp.mean(jnp.exp(1j * phases))
    circular_variance = 1 - jnp.abs(mean_phase_vector)
    
    # Phase concentration (von Mises concentration parameter estimate)
    R = jnp.abs(mean_phase_vector)
    if R < 0.85:
        concentration = 2 * R + R**3 + 5 * R**5 / 6
    else:
        concentration = 1 / (2 * (1 - R))
    
    return {
        'phase_variance': float(phase_var),
        'circular_variance': float(circular_variance),
        'mean_resultant_length': float(R),
        'concentration_parameter': float(concentration),
        'phase_range': float(jnp.max(phases) - jnp.min(phases))
    }


def spectral_analysis(x: Array) -> Dict[str, float]:
    """Analyze spectral properties of complex activations.
    
    Args:
        x: Complex tensor
        
    Returns:
        Dictionary with spectral metrics
    """
    # Flatten tensor for analysis
    x_flat = x.flatten()
    
    # Compute power spectral density
    fft_vals = jnp.fft.fft(x_flat)
    power_spectrum = jnp.abs(fft_vals) ** 2
    
    # Spectral centroid (center of mass of spectrum)
    freqs = jnp.fft.fftfreq(len(x_flat))
    spectral_centroid = jnp.sum(freqs * power_spectrum) / jnp.sum(power_spectrum)
    
    # Spectral spread (variance around centroid)
    spectral_spread = jnp.sqrt(
        jnp.sum((freqs - spectral_centroid) ** 2 * power_spectrum) / jnp.sum(power_spectrum)
    )
    
    # High frequency energy ratio
    nyquist = len(x_flat) // 2
    high_freq_power = jnp.sum(power_spectrum[nyquist//2:nyquist])
    total_power = jnp.sum(power_spectrum[:nyquist])
    high_freq_ratio = high_freq_power / (total_power + 1e-8)
    
    return {
        'spectral_centroid': float(spectral_centroid),
        'spectral_spread': float(spectral_spread),
        'high_frequency_ratio': float(high_freq_ratio),
        'total_power': float(total_power)
    }


def comprehensive_analysis(
    model_fn: callable,
    params: Dict,
    x_batch: Array,
    activation_fns: Optional[Dict[str, callable]] = None
) -> Dict[str, Any]:
    """Perform comprehensive analysis of complex neural network.
    
    Args:
        model_fn: Model function
        params: Model parameters
        x_batch: Batch of input data
        activation_fns: Dictionary of activation functions to analyze
        
    Returns:
        Dictionary with all analysis results
    """
    results = {}
    
    # Magnitude drift analysis
    results['magnitude_drift'] = track_magnitude_evolution(model_fn, params, x_batch)
    
    # Single sample detailed analysis
    x_single = x_batch[0:1]
    detailed_metrics = analyze_magnitude_drift(model_fn, params, x_single)
    results['detailed_metrics'] = detailed_metrics.to_dict()
    
    # Forward pass for activation analysis
    output, aux_data = model_fn(params, x_single)
    
    if 'activations' in aux_data:
        # Phase coherence analysis for each layer
        results['phase_coherence'] = {}
        results['spectral_analysis'] = {}
        
        for i, activation in enumerate(aux_data['activations']):
            results['phase_coherence'][f'layer_{i}'] = phase_coherence_analysis(activation)
            results['spectral_analysis'][f'layer_{i}'] = spectral_analysis(activation)
    
    # Holomorphicity analysis for activation functions
    if activation_fns is not None:
        results['holomorphicity_violations'] = {}
        
        # Create test points in complex plane
        x_test = jnp.linspace(-2, 2, 50)
        y_test = jnp.linspace(-2, 2, 50)
        X, Y = jnp.meshgrid(x_test, y_test)
        z_test = X + 1j * Y
        z_test = z_test.flatten()
        
        for name, activation_fn in activation_fns.items():
            violation = compute_holomorphicity_violation(activation_fn, z_test)
            results['holomorphicity_violations'][name] = violation
    
    return results