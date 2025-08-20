"""Unit tests for complex activation functions."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from holomorphic_networks.activations import (
    holomorphic_elu, holomorphic_swish, crelu, modrelu, complex_tanh,
    zrelu, cardioid, complex_sigmoid, get_activation, ACTIVATIONS
)
from holomorphic_networks.analysis import compute_holomorphicity_violation


class TestActivationFunctions:
    """Test suite for activation functions."""
    
    @pytest.fixture
    def complex_inputs(self):
        """Generate test complex inputs."""
        key = random.PRNGKey(42)
        # Various test cases
        return {
            'simple': jnp.array([1.0 + 1.0j, -1.0 + 1.0j, 1.0 - 1.0j, -1.0 - 1.0j]),
            'zeros': jnp.array([0.0 + 0.0j, 0.0 + 1.0j, 1.0 + 0.0j]),
            'large': jnp.array([10.0 + 10.0j, -10.0 + 10.0j, 10.0 - 10.0j]),
            'random': random.normal(key, (100,), dtype=jnp.complex64),
            'grid': self._create_complex_grid()
        }
    
    def _create_complex_grid(self):
        """Create a grid of complex numbers for testing."""
        x = jnp.linspace(-2, 2, 10)
        y = jnp.linspace(-2, 2, 10)
        X, Y = jnp.meshgrid(x, y)
        return (X + 1j * Y).flatten()
    
    def test_holomorphic_elu_properties(self, complex_inputs):
        """Test holomorphic ELU properties."""
        for name, z in complex_inputs.items():
            result = holomorphic_elu(z)
            
            # Check output is complex
            assert jnp.iscomplexobj(result)
            assert result.shape == z.shape
            
            # Check continuity at Re(z) = 0
            boundary_points = jnp.array([0.0 + 1.0j, 0.0 - 1.0j, 0.0 + 0.0j])
            boundary_result = holomorphic_elu(boundary_points)
            assert jnp.all(jnp.isfinite(boundary_result))
            
            # Check behavior for positive real part
            positive_real = jnp.array([1.0 + 1.0j, 2.0 + 0.5j])
            pos_result = holomorphic_elu(positive_real)
            # Should be identity for positive real part
            assert jnp.allclose(pos_result, positive_real, atol=1e-6)
    
    def test_holomorphic_swish_properties(self, complex_inputs):
        """Test holomorphic Swish properties."""
        for name, z in complex_inputs.items():
            result = holomorphic_swish(z)
            
            # Check output is complex
            assert jnp.iscomplexobj(result)
            assert result.shape == z.shape
            
            # Check finite output
            assert jnp.all(jnp.isfinite(result))
            
            # Check zero input gives zero output
            zero_result = holomorphic_swish(jnp.array([0.0 + 0.0j]))
            assert jnp.allclose(zero_result, 0.0, atol=1e-6)
    
    def test_crelu_properties(self, complex_inputs):
        """Test CReLU properties."""
        for name, z in complex_inputs.items():
            result = crelu(z)
            
            # Check output is complex
            assert jnp.iscomplexobj(result)
            assert result.shape == z.shape
            
            # Check real and imaginary parts are non-negative
            assert jnp.all(result.real >= 0)
            assert jnp.all(result.imag >= 0)
            
            # Check specific cases
            test_input = jnp.array([-1.0 - 1.0j, 1.0 + 1.0j, -1.0 + 1.0j])
            expected = jnp.array([0.0 + 0.0j, 1.0 + 1.0j, 0.0 + 1.0j])
            assert jnp.allclose(crelu(test_input), expected, atol=1e-6)
    
    def test_modrelu_properties(self, complex_inputs):
        """Test ModReLU properties."""
        for name, z in complex_inputs.items():
            result = modrelu(z)
            
            # Check output is complex
            assert jnp.iscomplexobj(result)
            assert result.shape == z.shape
            
            # Check magnitude is non-negative
            magnitudes = jnp.abs(result)
            assert jnp.all(magnitudes >= 0)
            
            # Check phase preservation for non-zero inputs
            non_zero_mask = jnp.abs(z) > 1e-8
            if jnp.any(non_zero_mask):
                input_phases = jnp.angle(z[non_zero_mask])
                output_phases = jnp.angle(result[non_zero_mask])
                # Phases should be similar (allowing for numerical precision)
                phase_diff = jnp.abs(jnp.exp(1j * (input_phases - output_phases)) - 1)
                assert jnp.all(phase_diff < 1e-5)
    
    def test_complex_tanh_properties(self, complex_inputs):
        """Test complex tanh properties."""
        for name, z in complex_inputs.items():
            result = complex_tanh(z)
            
            # Check output is complex
            assert jnp.iscomplexobj(result)
            assert result.shape == z.shape
            
            # Check bounded output (magnitude should be <= 1)
            magnitudes = jnp.abs(result)
            assert jnp.all(magnitudes <= 1.0 + 1e-6)  # Small tolerance for numerical precision
            
            # Check zero input gives zero output
            zero_result = complex_tanh(jnp.array([0.0 + 0.0j]))
            assert jnp.allclose(zero_result, 0.0, atol=1e-6)
    
    def test_zrelu_properties(self, complex_inputs):
        """Test zReLU properties."""
        for name, z in complex_inputs.items():
            result = zrelu(z)
            
            # Check output is complex
            assert jnp.iscomplexobj(result)
            assert result.shape == z.shape
            
            # Check specific behavior
            test_cases = jnp.array([
                1.0 + 1.0j,   # Both positive -> should pass through
                -1.0 + 1.0j,  # Real negative -> should be zero
                1.0 - 1.0j,   # Imag negative -> should be zero
                -1.0 - 1.0j   # Both negative -> should be zero
            ])
            expected = jnp.array([1.0 + 1.0j, 0.0, 0.0, 0.0])
            assert jnp.allclose(zrelu(test_cases), expected, atol=1e-6)
    
    def test_gradient_computation(self, complex_inputs):
        """Test that gradients can be computed for all activations."""
        def test_gradient(activation_fn, z):
            def loss_fn(x):
                return jnp.mean(jnp.abs(activation_fn(x)) ** 2)
            
            grad_fn = jax.grad(loss_fn)
            grad = grad_fn(z)
            
            # Check gradient is finite and has correct shape
            assert grad.shape == z.shape
            assert jnp.all(jnp.isfinite(grad))
            
            return grad
        
        z = complex_inputs['simple']
        
        # Test all activation functions
        activation_fns = [
            holomorphic_elu, holomorphic_swish, crelu, modrelu,
            complex_tanh, zrelu, complex_sigmoid
        ]
        
        for activation_fn in activation_fns:
            grad = test_gradient(activation_fn, z)
            assert grad is not None
    
    def test_holomorphicity_violations(self):
        """Test holomorphicity violations for activation functions."""
        # Create test grid
        x = jnp.linspace(-1, 1, 20)
        y = jnp.linspace(-1, 1, 20)
        X, Y = jnp.meshgrid(x, y)
        z_test = (X + 1j * Y).flatten()
        
        # Test holomorphic functions (should have low violations)
        holomorphic_fns = {
            'h_elu': holomorphic_elu,
            'h_swish': holomorphic_swish,
            'tanh': complex_tanh,
            'sigmoid': complex_sigmoid
        }
        
        for name, fn in holomorphic_fns.items():
            violation = compute_holomorphicity_violation(fn, z_test)
            # Holomorphic functions should have very low violation
            assert violation < 1e-3, f"{name} has high holomorphicity violation: {violation}"
        
        # Test non-holomorphic functions (should have higher violations)
        non_holomorphic_fns = {
            'crelu': crelu,
            'modrelu': modrelu,
            'zrelu': zrelu
        }
        
        for name, fn in non_holomorphic_fns.items():
            violation = compute_holomorphicity_violation(fn, z_test)
            # Non-holomorphic functions should have higher violations
            assert violation > 1e-6, f"{name} unexpectedly has low violation: {violation}"
    
    def test_activation_registry(self):
        """Test activation function registry."""
        # Test that all registered activations work
        for name in ACTIVATIONS.keys():
            activation_fn = get_activation(name)
            assert callable(activation_fn)
            
            # Test with simple input
            test_input = jnp.array([1.0 + 1.0j, -1.0 + 1.0j])
            result = activation_fn(test_input)
            assert result.shape == test_input.shape
            assert jnp.iscomplexobj(result)
        
        # Test invalid activation name
        with pytest.raises(ValueError):
            get_activation('invalid_activation')
    
    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        # Very large values
        large_vals = jnp.array([100.0 + 100.0j, -100.0 + 100.0j])
        
        # Very small values
        small_vals = jnp.array([1e-10 + 1e-10j, -1e-10 + 1e-10j])
        
        # Test that functions don't produce NaN or inf
        test_fns = [holomorphic_elu, holomorphic_swish, complex_tanh]
        
        for fn in test_fns:
            # Large values
            result_large = fn(large_vals)
            assert jnp.all(jnp.isfinite(result_large)), f"{fn.__name__} produced non-finite values for large inputs"
            
            # Small values
            result_small = fn(small_vals)
            assert jnp.all(jnp.isfinite(result_small)), f"{fn.__name__} produced non-finite values for small inputs"
    
    def test_dtype_preservation(self):
        """Test that activation functions preserve input dtype."""
        test_input_64 = jnp.array([1.0 + 1.0j, -1.0 + 1.0j], dtype=jnp.complex64)
        test_input_128 = jnp.array([1.0 + 1.0j, -1.0 + 1.0j], dtype=jnp.complex128)
        
        activation_fns = [holomorphic_elu, holomorphic_swish, complex_tanh]
        
        for fn in activation_fns:
            result_64 = fn(test_input_64)
            result_128 = fn(test_input_128)
            
            assert result_64.dtype == jnp.complex64, f"{fn.__name__} changed dtype from complex64"
            assert result_128.dtype == jnp.complex128, f"{fn.__name__} changed dtype from complex128"
    
    def test_vectorization(self):
        """Test that activation functions work with different array shapes."""
        shapes = [(10,), (5, 5), (2, 3, 4), (1, 1, 1, 10)]
        
        for shape in shapes:
            key = random.PRNGKey(42)
            test_input = random.normal(key, shape, dtype=jnp.complex64)
            
            for name, fn in ACTIVATIONS.items():
                result = fn(test_input)
                assert result.shape == test_input.shape, f"{name} changed shape from {test_input.shape} to {result.shape}"
                assert jnp.iscomplexobj(result), f"{name} returned non-complex result"


if __name__ == "__main__":
    pytest.main([__file__])