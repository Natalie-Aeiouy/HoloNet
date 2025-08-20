#!/usr/bin/env python3
"""
Simple debug script for gradient estimation methods.
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from holomorphic_networks.gradient_estimation import (
    GradientEstimator, 
    GradientEstimationConfig
)

def test_simple_exponential():
    """Test FFT gradient estimation on simple exponential."""
    print("Testing simple exponential: z(t) = exp(i*t)")
    
    # Simple configuration
    config = GradientEstimationConfig(
        method="fft",
        max_frequencies=25,
        talbot_n=12
    )
    estimator = GradientEstimator(config)
    
    # Simple test case: z(t) = exp(i*t), dz/dt = i*exp(i*t)
    T = 64  # Power of 2 for FFT efficiency
    times = jnp.linspace(0, 2*jnp.pi, T, endpoint=False)  # One full period
    
    trajectory = jnp.exp(1j * times)
    true_gradient = 1j * jnp.exp(1j * times)
    
    print(f"Time range: [0, {times[-1]:.3f}]")
    print(f"Number of points: {T}")
    print(f"dt: {(times[1] - times[0]):.6f}")
    
    # Test FFT method only
    try:
        estimated_gradient = estimator.fft_gradient(trajectory, times)
        
        # Compute error metrics
        error = jnp.abs(estimated_gradient - true_gradient)
        max_error = jnp.max(error)
        mean_error = jnp.mean(error)
        relative_error = mean_error / jnp.mean(jnp.abs(true_gradient))
        
        print(f"Max error: {max_error:.6f}")
        print(f"Mean error: {mean_error:.6f}")
        print(f"Relative error: {relative_error:.6f}")
        print(f"Test {'PASSED' if relative_error < 0.01 else 'FAILED'}")
        
        # Plot comparison for first few points
        print("\nFirst 10 points comparison:")
        print("Index | True Gradient | Estimated | Error")
        print("-" * 50)
        for i in range(min(10, T)):
            true_val = true_gradient[i]
            est_val = estimated_gradient[i]
            err = error[i]
            print(f"{i:5d} | {true_val:.6f} | {est_val:.6f} | {err:.6f}")
        
        return relative_error < 0.01
        
    except Exception as e:
        print(f"FFT gradient estimation failed: {e}")
        return False

def test_simple_polynomial():
    """Test on polynomial: z(t) = t^2"""
    print("\nTesting polynomial: z(t) = t^2, dz/dt = 2*t")
    
    config = GradientEstimationConfig(method="fft", max_frequencies=25)
    estimator = GradientEstimator(config)
    
    T = 64
    times = jnp.linspace(0.1, 3.0, T)  # Avoid t=0
    
    trajectory = times**2
    true_gradient = 2 * times
    
    try:
        estimated_gradient = estimator.fft_gradient(trajectory, times)
        
        error = jnp.abs(estimated_gradient - true_gradient)
        mean_error = jnp.mean(error)
        relative_error = mean_error / jnp.mean(jnp.abs(true_gradient))
        
        print(f"Relative error: {relative_error:.6f}")
        print(f"Test {'PASSED' if relative_error < 0.1 else 'FAILED'}")
        
        return relative_error < 0.1
        
    except Exception as e:
        print(f"Polynomial test failed: {e}")
        return False

def debug_fft_internals():
    """Debug FFT gradient computation step by step."""
    print("\nDebugging FFT internals...")
    
    # Simple sine wave
    T = 32
    times = jnp.linspace(0, 2*jnp.pi, T, endpoint=False)
    omega = 1.0
    trajectory = jnp.sin(omega * times)
    true_gradient = omega * jnp.cos(omega * times)
    
    print(f"Input trajectory shape: {trajectory.shape}")
    print(f"Input trajectory dtype: {trajectory.dtype}")
    
    # Manual FFT gradient computation
    dt = times[1] - times[0]
    print(f"dt: {dt:.6f}")
    
    # FFT
    traj_fft = jnp.fft.fft(trajectory)
    freqs = jnp.fft.fftfreq(T, dt)
    omega_vals = 2j * jnp.pi * freqs
    
    print(f"FFT shape: {traj_fft.shape}")
    print(f"Frequencies shape: {freqs.shape}")
    print(f"First 5 frequencies: {freqs[:5]}")
    print(f"First 5 omega values: {omega_vals[:5]}")
    
    # Compute gradient in frequency domain
    grad_fft = omega_vals * traj_fft
    print(f"Gradient FFT shape: {grad_fft.shape}")
    
    # IFFT
    grad_est = jnp.fft.ifft(grad_fft)
    grad_est_real = jnp.real(grad_est)  # Should be real for sine input
    
    print(f"Estimated gradient shape: {grad_est.shape}")
    print(f"Estimated gradient (real part): {grad_est_real[:5]}")
    print(f"True gradient: {true_gradient[:5]}")
    
    error = jnp.abs(grad_est_real - true_gradient)
    print(f"Error: {error[:5]}")
    print(f"Mean error: {jnp.mean(error):.6f}")

def main():
    """Run debug tests."""
    print("=" * 50)
    print("GRADIENT ESTIMATION DEBUG")
    print("=" * 50)
    
    # Test 1: Simple exponential
    success1 = test_simple_exponential()
    
    # Test 2: Simple polynomial  
    success2 = test_simple_polynomial()
    
    # Debug FFT internals
    debug_fft_internals()
    
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Exponential test: {'PASSED' if success1 else 'FAILED'}")
    print(f"Polynomial test: {'PASSED' if success2 else 'FAILED'}")
    
    if success1 and success2:
        print("✅ Basic gradient estimation is working!")
    else:
        print("❌ Issues detected in gradient estimation")

if __name__ == "__main__":
    main()