"""
Gradient Estimation Methods for Fourier Neural ODEs (FNODEs)

This module implements advanced gradient estimation techniques that enable
simulation-free training by directly matching vector fields in frequency space.

Key Methods:
1. FFT-based gradient estimation for periodic signals
2. Laplace transform integration using Talbot's method for transient signals  
3. Hybrid approach with automatic signal decomposition
4. FNODE loss functions for gradient flow matching

References:
- "Fourier Neural ODEs" - FFT-based gradient matching
- Talbot (1979) - "The accurate numerical inversion of the Laplace transform"
- Abate & Whitt (2006) - "A unified framework for numerically inverting Laplace transforms"
"""

from typing import Tuple, Optional, Callable, Union, Dict, Any
import jax
import jax.numpy as jnp
from jax import Array
from functools import partial
import numpy as np
from dataclasses import dataclass

# Type aliases
ComplexArray = Array  # Complex-valued JAX array
RealArray = Array     # Real-valued JAX array

@dataclass
class GradientEstimationConfig:
    """Configuration for gradient estimation methods."""
    method: str = "fft"  # "fft", "laplace", "hybrid"
    max_frequencies: int = 50  # K parameter for frequency truncation
    talbot_n: int = 24  # Number of Talbot contour points
    sigma: float = 1.0  # Talbot contour parameter
    noise_threshold: float = 1e-6  # Threshold for signal/noise separation
    hybrid_split_ratio: float = 0.5  # Ratio for hybrid decomposition

class GradientEstimator:
    """
    Advanced gradient estimation for Fourier Neural ODEs.
    
    Implements multiple methods for estimating temporal gradients from trajectory data,
    enabling simulation-free training by matching vector fields in frequency domain.
    """
    
    def __init__(self, config: GradientEstimationConfig):
        self.config = config
        
    def estimate_gradient(
        self, 
        trajectory: ComplexArray, 
        times: RealArray,
        method: Optional[str] = None
    ) -> ComplexArray:
        """
        Estimate temporal gradient dz/dt from trajectory data.
        
        Args:
            trajectory: Complex trajectory data [T, ...] where T is time steps
            times: Time points corresponding to trajectory samples [T]
            method: Override config method ("fft", "laplace", "hybrid")
            
        Returns:
            Estimated gradients dz/dt with same shape as trajectory
        """
        method = method or self.config.method
        
        if method == "fft":
            return self.fft_gradient(trajectory, times)
        elif method == "laplace":
            return self.laplace_gradient(trajectory, times)
        elif method == "hybrid":
            return self.hybrid_gradient(trajectory, times)
        else:
            raise ValueError(f"Unknown gradient estimation method: {method}")
    
    @partial(jax.jit, static_argnames=['self'])
    def fft_gradient(
        self, 
        trajectory: ComplexArray, 
        times: RealArray
    ) -> ComplexArray:
        """
        FFT-based gradient estimation for periodic signals.
        
        Estimates dz/dt via Fourier transform:
        dz/dt = IFFT[i*ω * FFT(z)]
        
        Args:
            trajectory: Complex trajectory [T, ...] 
            times: Time points [T]
            
        Returns:
            Estimated gradients [T, ...]
        """
        # Ensure uniform time sampling for FFT
        T = len(times)
        dt = (times[-1] - times[0]) / (T - 1)
        
        # Apply FFT along time axis
        trajectory_fft = jnp.fft.fft(trajectory, axis=0)
        
        # Create frequency grid
        freqs = jnp.fft.fftfreq(T, dt)
        omega = 2j * jnp.pi * freqs
        
        # Apply frequency truncation (keep only K frequencies)
        K = min(self.config.max_frequencies, T // 2)
        
        # Create truncated omega with proper shape handling
        if trajectory.ndim == 1:
            omega_truncated = jnp.zeros_like(omega)
            omega_truncated = omega_truncated.at[:K].set(omega[:K])
            if K > 0:
                omega_truncated = omega_truncated.at[-K:].set(omega[-K:])
            
            # Compute gradient in frequency domain
            gradient_fft = omega_truncated * trajectory_fft
        else:
            # Handle multi-dimensional arrays
            omega_truncated = jnp.zeros_like(omega)
            omega_truncated = omega_truncated.at[:K].set(omega[:K])
            if K > 0:
                omega_truncated = omega_truncated.at[-K:].set(omega[-K:])
            
            # Expand omega for broadcasting
            shape = [1] * trajectory.ndim
            shape[0] = T
            omega_expanded = omega_truncated.reshape(shape)
            
            # Compute gradient in frequency domain
            gradient_fft = omega_expanded * trajectory_fft
        
        # Transform back to time domain
        gradient = jnp.fft.ifft(gradient_fft, axis=0)
        
        return gradient
    
    def laplace_gradient(
        self, 
        trajectory: ComplexArray, 
        times: RealArray
    ) -> ComplexArray:
        """
        Laplace transform-based gradient estimation for transient signals.
        
        Uses Talbot's method for numerical inverse Laplace transform:
        dz/dt = L^{-1}[s * L(z)]
        
        Args:
            trajectory: Complex trajectory [T, ...]
            times: Time points [T]
            
        Returns:
            Estimated gradients [T, ...]
        """
        # Generate Talbot contour points
        contour_points, weights = self._generate_talbot_contour(times)
        
        # Compute Laplace transform at contour points
        laplace_values = self._compute_laplace_transform(
            trajectory, times, contour_points
        )
        
        # Multiply by s for derivative
        laplace_derivative = contour_points[:, None] * laplace_values
        
        # Inverse transform using Talbot quadrature
        gradient = self._inverse_laplace_talbot(
            laplace_derivative, weights, times
        )
        
        return gradient
    
    def hybrid_gradient(
        self, 
        trajectory: ComplexArray, 
        times: RealArray
    ) -> ComplexArray:
        """
        Hybrid approach: decompose signal into periodic/transient components.
        
        1. Separate periodic and transient components
        2. Apply FFT to periodic part, Laplace to transient part
        3. Combine results
        
        Args:
            trajectory: Complex trajectory [T, ...]
            times: Time points [T]
            
        Returns:
            Estimated gradients [T, ...]
        """
        # Decompose signal (simplified approach)
        periodic_part, transient_part = self._decompose_signal(trajectory, times)
        
        # Apply appropriate method to each component
        grad_periodic = self.fft_gradient(periodic_part, times)
        grad_transient = self.laplace_gradient(transient_part, times)
        
        # Combine with weighted sum
        ratio = self.config.hybrid_split_ratio
        gradient = ratio * grad_periodic + (1 - ratio) * grad_transient
        
        return gradient
    
    def _generate_talbot_contour(self, times: RealArray) -> Tuple[ComplexArray, ComplexArray]:
        """
        Generate Talbot contour points and weights for inverse Laplace transform.
        
        Based on Talbot (1979) optimal contour for numerical inversion.
        
        Args:
            times: Time points for inversion [T]
            
        Returns:
            contour_points: Complex contour points [N]
            weights: Quadrature weights [N]
        """
        N = self.config.talbot_n
        
        # Adaptive sigma based on time range for better numerical stability
        t_max = jnp.max(times)
        sigma = jnp.maximum(0.5 / t_max, 0.1)  # Ensure sigma is reasonable
        
        # Talbot contour parameters
        k = jnp.arange(1, N + 1)
        theta = k * jnp.pi / N
        
        # Avoid division by zero near theta = 0
        theta = jnp.where(theta < 1e-10, 1e-10, theta)
        
        # Contour points - use stable cotangent computation
        cot_theta = jnp.cos(theta) / jnp.sin(theta)
        z = sigma + 1j * theta * (cot_theta + 1j)
        
        # Quadrature weights with numerical stability
        sin_theta_sq = jnp.sin(theta)**2
        sin_theta_sq = jnp.where(sin_theta_sq < 1e-10, 1e-10, sin_theta_sq)
        
        weights = 2 * sigma / N * (1 + 1j * theta * (1 / sin_theta_sq - 1))
        
        return z, weights
    
    def _compute_laplace_transform(
        self, 
        trajectory: ComplexArray, 
        times: RealArray, 
        contour_points: ComplexArray
    ) -> ComplexArray:
        """
        Compute Laplace transform of trajectory at given contour points.
        
        L(z)(s) = ∫₀^∞ z(t) * exp(-s*t) dt
        
        Args:
            trajectory: Complex trajectory [T, ...]
            times: Time points [T]
            contour_points: Complex contour points [N]
            
        Returns:
            Laplace transform values [N, ...]
        """
        # Numerical integration using trapezoidal rule
        dt = times[1] - times[0]  # Assume uniform sampling
        
        # Expand dimensions for broadcasting
        s = contour_points[:, None, ...]  # [N, 1, ...]
        t = times[None, :, ...]           # [1, T, ...]
        z = trajectory[None, :, ...]      # [1, T, ...]
        
        # Compute integrand: z(t) * exp(-s*t)
        integrand = z * jnp.exp(-s * t)
        
        # Integrate using trapezoidal rule
        laplace_values = jnp.trapz(integrand, dx=dt, axis=1)
        
        return laplace_values
    
    def _inverse_laplace_talbot(
        self, 
        laplace_values: ComplexArray, 
        weights: ComplexArray, 
        times: RealArray
    ) -> ComplexArray:
        """
        Perform inverse Laplace transform using Talbot quadrature.
        
        z(t) = (1/2πi) ∑ᵢ wᵢ * F(sᵢ) * exp(sᵢ*t)
        
        Args:
            laplace_values: Laplace transform values [N, ...]
            weights: Quadrature weights [N]
            times: Time points [T]
            
        Returns:
            Inverse transform [T, ...]
        """
        # Expand dimensions for broadcasting
        w = weights[:, None, ...]         # [N, 1, ...]
        F = laplace_values                # [N, ...]
        t = times[None, :, ...]           # [1, T, ...]
        s = self._generate_talbot_contour(times)[0][:, None, ...]  # [N, 1, ...]
        
        # Compute sum: ∑ᵢ wᵢ * F(sᵢ) * exp(sᵢ*t)
        summand = w * F[:, None, ...] * jnp.exp(s * t)
        result = jnp.sum(summand, axis=0)
        
        # Apply scaling factor
        result = jnp.real(result) / (2 * jnp.pi)
        
        return result.astype(laplace_values.dtype)
    
    def _decompose_signal(
        self, 
        trajectory: ComplexArray, 
        times: RealArray
    ) -> Tuple[ComplexArray, ComplexArray]:
        """
        Decompose signal into periodic and transient components.
        
        Simple approach: use frequency analysis to separate components.
        
        Args:
            trajectory: Complex trajectory [T, ...]
            times: Time points [T]
            
        Returns:
            periodic_part: Periodic component [T, ...]
            transient_part: Transient component [T, ...]
        """
        # FFT analysis
        trajectory_fft = jnp.fft.fft(trajectory, axis=0)
        freqs = jnp.fft.fftfreq(len(times))
        
        # Simple frequency-based separation
        # High magnitude frequencies -> periodic
        # Low magnitude frequencies -> transient
        magnitude_threshold = jnp.percentile(jnp.abs(trajectory_fft), 80)
        
        periodic_mask = jnp.abs(trajectory_fft) > magnitude_threshold
        transient_mask = ~periodic_mask
        
        # Reconstruct components
        periodic_fft = trajectory_fft * periodic_mask[:, None]
        transient_fft = trajectory_fft * transient_mask[:, None]
        
        periodic_part = jnp.fft.ifft(periodic_fft, axis=0)
        transient_part = jnp.fft.ifft(transient_fft, axis=0)
        
        return periodic_part, transient_part


# FNODE Loss Functions
def fourier_gradient_matching_loss(
    predicted_gradients: ComplexArray,
    estimated_gradients: ComplexArray,
    frequencies: Optional[RealArray] = None,
    weight_by_frequency: bool = True
) -> float:
    """
    FNODE loss function for gradient flow matching.
    
    L = (1/N) ∑ₙ ||dz/dt_pred(tₙ) - dz/dt_est(tₙ)||²
    
    Args:
        predicted_gradients: Network predicted gradients [T, ...]
        estimated_gradients: Estimated gradients from data [T, ...]
        frequencies: Frequency weights [T] (optional)
        weight_by_frequency: Whether to weight loss by frequency importance
        
    Returns:
        Scalar loss value
    """
    # Compute pointwise squared error
    squared_error = jnp.abs(predicted_gradients - estimated_gradients)**2
    
    if weight_by_frequency and frequencies is not None:
        # Weight by frequency importance (higher frequencies less important)
        weights = 1.0 / (1.0 + jnp.abs(frequencies))
        weights = weights / jnp.sum(weights)  # Normalize
        weighted_error = squared_error * weights[:, None]
        loss = jnp.sum(weighted_error)
    else:
        loss = jnp.mean(squared_error)
    
    return loss


def laplace_gradient_matching_loss(
    predicted_gradients: ComplexArray,
    estimated_gradients: ComplexArray,
    decay_weights: Optional[RealArray] = None
) -> float:
    """
    Laplace-based loss function for transient signal matching.
    
    Args:
        predicted_gradients: Network predicted gradients [T, ...]
        estimated_gradients: Estimated gradients from data [T, ...]
        decay_weights: Exponential decay weights [T] (optional)
        
    Returns:
        Scalar loss value
    """
    squared_error = jnp.abs(predicted_gradients - estimated_gradients)**2
    
    if decay_weights is not None:
        weighted_error = squared_error * decay_weights[:, None]
        loss = jnp.sum(weighted_error) / jnp.sum(decay_weights)
    else:
        loss = jnp.mean(squared_error)
    
    return loss


# Utility functions
def validate_gradient_estimation(
    estimator: GradientEstimator,
    analytical_function: Callable[[RealArray], ComplexArray],
    analytical_derivative: Callable[[RealArray], ComplexArray],
    times: RealArray,
    rtol: float = 1e-2
) -> Dict[str, float]:
    """
    Validate gradient estimation accuracy against analytical solutions.
    
    Args:
        estimator: Gradient estimator instance
        analytical_function: True function z(t)
        analytical_derivative: True derivative dz/dt
        times: Time points for validation
        rtol: Relative tolerance for validation
        
    Returns:
        Dictionary with validation metrics
    """
    # Generate synthetic trajectory
    trajectory = analytical_function(times)
    true_gradient = analytical_derivative(times)
    
    # Estimate gradients using different methods
    methods = ["fft", "laplace", "hybrid"]
    results = {}
    
    for method in methods:
        try:
            estimated_gradient = estimator.estimate_gradient(trajectory, times, method)
            error = jnp.mean(jnp.abs(estimated_gradient - true_gradient))
            relative_error = error / jnp.mean(jnp.abs(true_gradient))
            
            results[f"{method}_error"] = float(error)
            results[f"{method}_relative_error"] = float(relative_error)
            results[f"{method}_passes"] = float(relative_error) < rtol
            
        except Exception as e:
            results[f"{method}_error"] = float('inf')
            results[f"{method}_relative_error"] = float('inf')
            results[f"{method}_passes"] = False
    
    return results


# Example usage and testing functions
def test_gradient_estimation():
    """Test gradient estimation on known analytical functions."""
    
    # Test configuration
    config = GradientEstimationConfig(
        method="hybrid",
        max_frequencies=25,
        talbot_n=24
    )
    estimator = GradientEstimator(config)
    
    # Test on exponential decay: z(t) = exp(-αt) * exp(iωt)
    alpha, omega = 0.1, 2.0
    
    def analytical_fn(t):
        return jnp.exp(-alpha * t) * jnp.exp(1j * omega * t)
    
    def analytical_derivative(t):
        return (-alpha + 1j * omega) * jnp.exp(-alpha * t) * jnp.exp(1j * omega * t)
    
    # Time grid
    T = 100
    times = jnp.linspace(0, 10, T)
    
    # Validate estimation
    metrics = validate_gradient_estimation(
        estimator, analytical_fn, analytical_derivative, times
    )
    
    print("Gradient Estimation Validation:")
    for method in ["fft", "laplace", "hybrid"]:
        error = metrics[f"{method}_relative_error"]
        passes = metrics[f"{method}_passes"]
        print(f"  {method:8}: {error:.6f} {'✓' if passes else '✗'}")
    
    return metrics


if __name__ == "__main__":
    # Run validation tests
    test_gradient_estimation()