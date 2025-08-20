#!/usr/bin/env python3
"""
Test and benchmark suite for gradient estimation methods.

This script validates the accuracy of FFT, Laplace, and hybrid gradient estimation
methods against analytical solutions, providing comprehensive performance metrics.
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Callable
import time

from holomorphic_networks.gradient_estimation import (
    GradientEstimator, 
    GradientEstimationConfig,
    validate_gradient_estimation,
    fourier_gradient_matching_loss,
    laplace_gradient_matching_loss
)

# Enable 64-bit precision for better numerical accuracy
jax.config.update("jax_enable_x64", True)

class GradientEstimationBenchmark:
    """Comprehensive benchmark suite for gradient estimation methods."""
    
    def __init__(self):
        self.config = GradientEstimationConfig(
            method="hybrid",
            max_frequencies=50,
            talbot_n=24,
            sigma=1.0
        )
        self.estimator = GradientEstimator(self.config)
    
    def test_exponential_decay(self) -> Dict[str, float]:
        """Test on exponential decay: z(t) = exp(-αt + iωt)"""
        print("🧪 Testing Exponential Decay Function")
        
        alpha, omega = 0.2, 3.0
        T = 200
        times = jnp.linspace(0, 15, T)
        
        def analytical_fn(t):
            return jnp.exp(-alpha * t + 1j * omega * t)
        
        def analytical_derivative(t):
            return (-alpha + 1j * omega) * jnp.exp(-alpha * t + 1j * omega * t)
        
        metrics = validate_gradient_estimation(
            self.estimator, analytical_fn, analytical_derivative, times, rtol=0.05
        )
        
        self._print_metrics("Exponential Decay", metrics)
        return metrics
    
    def test_polynomial_complex(self) -> Dict[str, float]:
        """Test on complex polynomial: z(t) = (t + it)³"""
        print("🧪 Testing Complex Polynomial Function")
        
        T = 150
        times = jnp.linspace(0.1, 8, T)  # Avoid t=0 for numerical stability
        
        def analytical_fn(t):
            z_t = t + 1j * t
            return z_t**3
        
        def analytical_derivative(t):
            z_t = t + 1j * t
            return 3 * z_t**2 * (1 + 1j)
        
        metrics = validate_gradient_estimation(
            self.estimator, analytical_fn, analytical_derivative, times, rtol=0.1
        )
        
        self._print_metrics("Complex Polynomial", metrics)
        return metrics
    
    def test_oscillatory_envelope(self) -> Dict[str, float]:
        """Test on modulated oscillation: z(t) = t²·exp(iωt)"""
        print("🧪 Testing Oscillatory Envelope Function")
        
        omega = 4.0
        T = 180
        times = jnp.linspace(0, 12, T)
        
        def analytical_fn(t):
            return t**2 * jnp.exp(1j * omega * t)
        
        def analytical_derivative(t):
            return 2*t * jnp.exp(1j * omega * t) + t**2 * 1j * omega * jnp.exp(1j * omega * t)
        
        metrics = validate_gradient_estimation(
            self.estimator, analytical_fn, analytical_derivative, times, rtol=0.08
        )
        
        self._print_metrics("Oscillatory Envelope", metrics)
        return metrics
    
    def test_damped_oscillator(self) -> Dict[str, float]:
        """Test on damped harmonic oscillator solution"""
        print("🧪 Testing Damped Harmonic Oscillator")
        
        # Parameters: ω₀ = 5, γ = 0.5 (underdamped)
        omega_0, gamma = 5.0, 0.5
        omega_d = jnp.sqrt(omega_0**2 - gamma**2)
        
        T = 250
        times = jnp.linspace(0, 10, T)
        
        def analytical_fn(t):
            # x(t) = exp(-γt) * [A*cos(ω_d*t) + B*sin(ω_d*t)]
            # For complex representation: z(t) = exp(-γt + iω_d*t)
            return jnp.exp(-gamma * t + 1j * omega_d * t)
        
        def analytical_derivative(t):
            return (-gamma + 1j * omega_d) * jnp.exp(-gamma * t + 1j * omega_d * t)
        
        metrics = validate_gradient_estimation(
            self.estimator, analytical_fn, analytical_derivative, times, rtol=0.06
        )
        
        self._print_metrics("Damped Oscillator", metrics)
        return metrics
    
    def test_chirp_signal(self) -> Dict[str, float]:
        """Test on frequency-swept signal: z(t) = exp(i(ω₀t + αt²/2))"""
        print("🧪 Testing Chirp Signal")
        
        omega_0, alpha = 2.0, 0.8  # Linear chirp parameters
        T = 200
        times = jnp.linspace(0, 10, T)
        
        def analytical_fn(t):
            phase = omega_0 * t + alpha * t**2 / 2
            return jnp.exp(1j * phase)
        
        def analytical_derivative(t):
            instantaneous_freq = omega_0 + alpha * t
            return 1j * instantaneous_freq * jnp.exp(1j * (omega_0 * t + alpha * t**2 / 2))
        
        metrics = validate_gradient_estimation(
            self.estimator, analytical_fn, analytical_derivative, times, rtol=0.1
        )
        
        self._print_metrics("Chirp Signal", metrics)
        return metrics
    
    def benchmark_performance(self) -> Dict[str, float]:
        """Benchmark computational performance of different methods."""
        print("⚡ Performance Benchmarking")
        
        # Setup test case
        alpha, omega = 0.15, 2.5
        T = 500
        times = jnp.linspace(0, 20, T)
        trajectory = jnp.exp(-alpha * times + 1j * omega * times)
        
        methods = ["fft", "laplace", "hybrid"]
        performance = {}
        
        # Warmup JIT compilation
        for method in methods:
            _ = self.estimator.estimate_gradient(trajectory, times, method)
        
        # Benchmark each method
        for method in methods:
            times_list = []
            
            for _ in range(10):  # Multiple runs for averaging
                start_time = time.time()
                _ = self.estimator.estimate_gradient(trajectory, times, method)
                end_time = time.time()
                times_list.append(end_time - start_time)
            
            avg_time = np.mean(times_list)
            std_time = np.std(times_list)
            performance[f"{method}_time_mean"] = avg_time
            performance[f"{method}_time_std"] = std_time
            
            print(f"  {method:8}: {avg_time:.4f} ± {std_time:.4f} seconds")
        
        return performance
    
    def test_noise_robustness(self) -> Dict[str, float]:
        """Test robustness to measurement noise."""
        print("🔊 Testing Noise Robustness")
        
        # Clean signal
        alpha, omega = 0.1, 3.0
        T = 200
        times = jnp.linspace(0, 15, T)
        clean_trajectory = jnp.exp(-alpha * times + 1j * omega * times)
        true_gradient = (-alpha + 1j * omega) * clean_trajectory
        
        noise_levels = [0.01, 0.05, 0.1, 0.2]
        methods = ["fft", "laplace", "hybrid"]
        
        results = {}
        
        for noise_level in noise_levels:
            print(f"  Noise level: {noise_level:.2f}")
            
            # Add complex Gaussian noise
            key = jax.random.PRNGKey(42)
            noise_real = jax.random.normal(key, shape=clean_trajectory.shape) * noise_level
            noise_imag = jax.random.normal(key, shape=clean_trajectory.shape) * noise_level
            noise = noise_real + 1j * noise_imag
            
            noisy_trajectory = clean_trajectory + noise
            
            for method in methods:
                estimated_gradient = self.estimator.estimate_gradient(
                    noisy_trajectory, times, method
                )
                
                error = jnp.mean(jnp.abs(estimated_gradient - true_gradient))
                relative_error = error / jnp.mean(jnp.abs(true_gradient))
                
                key = f"noise_{noise_level:.2f}_{method}_rel_error"
                results[key] = float(relative_error)
                
                print(f"    {method:8}: {relative_error:.4f}")
        
        return results
    
    def test_fnode_loss_functions(self) -> Dict[str, float]:
        """Test FNODE loss function implementations."""
        print("📊 Testing FNODE Loss Functions")
        
        # Generate test data
        T = 100
        times = jnp.linspace(0, 10, T)
        freqs = jnp.fft.fftfreq(T)
        
        # True gradients
        true_gradients = jnp.exp(1j * 2 * jnp.pi * times)
        
        # Simulated predictions with controlled error
        pred_gradients = true_gradients + 0.1 * jax.random.normal(
            jax.random.PRNGKey(42), true_gradients.shape, dtype=jnp.complex64
        )
        
        # Test Fourier loss
        fourier_loss = fourier_gradient_matching_loss(
            pred_gradients, true_gradients, freqs, weight_by_frequency=True
        )
        
        # Test Laplace loss
        decay_weights = jnp.exp(-0.1 * times)
        laplace_loss = laplace_gradient_matching_loss(
            pred_gradients, true_gradients, decay_weights
        )
        
        print(f"  Fourier loss:  {fourier_loss:.6f}")
        print(f"  Laplace loss:  {laplace_loss:.6f}")
        
        return {
            "fourier_loss": float(fourier_loss),
            "laplace_loss": float(laplace_loss)
        }
    
    def _print_metrics(self, test_name: str, metrics: Dict[str, float]):
        """Pretty print validation metrics."""
        print(f"  Results for {test_name}:")
        for method in ["fft", "laplace", "hybrid"]:
            error = metrics[f"{method}_relative_error"]
            passes = metrics[f"{method}_passes"]
            status = "✅ PASS" if passes else "❌ FAIL"
            print(f"    {method:8}: {error:.6f} {status}")
        print()
    
    def run_full_benchmark(self) -> Dict[str, Dict[str, float]]:
        """Run complete benchmark suite."""
        print("=" * 60)
        print("🚀 GRADIENT ESTIMATION BENCHMARK SUITE")
        print("=" * 60)
        print()
        
        results = {}
        
        # Accuracy tests
        results["exponential_decay"] = self.test_exponential_decay()
        results["polynomial_complex"] = self.test_polynomial_complex()
        results["oscillatory_envelope"] = self.test_oscillatory_envelope()
        results["damped_oscillator"] = self.test_damped_oscillator()
        results["chirp_signal"] = self.test_chirp_signal()
        
        # Performance tests
        results["performance"] = self.benchmark_performance()
        results["noise_robustness"] = self.test_noise_robustness()
        results["fnode_losses"] = self.test_fnode_loss_functions()
        
        # Summary
        print("=" * 60)
        print("📊 BENCHMARK SUMMARY")
        print("=" * 60)
        
        # Count passes across all accuracy tests
        total_tests = 0
        passed_tests = 0
        
        accuracy_tests = ["exponential_decay", "polynomial_complex", "oscillatory_envelope", 
                         "damped_oscillator", "chirp_signal"]
        
        for test in accuracy_tests:
            for method in ["fft", "laplace", "hybrid"]:
                total_tests += 1
                if results[test][f"{method}_passes"]:
                    passed_tests += 1
        
        pass_rate = passed_tests / total_tests * 100
        print(f"Overall Pass Rate: {passed_tests}/{total_tests} ({pass_rate:.1f}%)")
        
        # Performance summary
        perf = results["performance"]
        fastest_method = min(["fft", "laplace", "hybrid"], 
                           key=lambda m: perf[f"{m}_time_mean"])
        print(f"Fastest Method: {fastest_method} ({perf[f'{fastest_method}_time_mean']:.4f}s)")
        
        print("\n🎉 Gradient estimation benchmark complete!")
        
        return results


def main():
    """Run the gradient estimation benchmark."""
    benchmark = GradientEstimationBenchmark()
    results = benchmark.run_full_benchmark()
    
    # Optional: Save results to file
    import json
    with open("gradient_estimation_benchmark_results.json", "w") as f:
        # Convert numpy types to native Python types for JSON serialization
        json_results = {}
        for test_name, test_results in results.items():
            json_results[test_name] = {
                k: float(v) if isinstance(v, (np.ndarray, jnp.ndarray)) else v
                for k, v in test_results.items()
            }
        json.dump(json_results, f, indent=2)
    
    print("📁 Results saved to gradient_estimation_benchmark_results.json")
    
    return results


if __name__ == "__main__":
    main()