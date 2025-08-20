# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HoloNet is a research implementation of holomorphic complex-valued neural networks using JAX. The project explores novel activation functions that maintain complex differentiability while avoiding magnitude drift in deep networks.

## Development Environment

- **Python Version**: 3.13.7
- **Virtual Environment**: `nvenv/` (managed with uv)
- **Framework**: JAX with JIT compilation

## Common Development Commands

### Environment Setup
```bash
# Activate virtual environment
source nvenv/bin/activate

# Install dependencies (after creating requirements.txt)
pip install -r requirements.txt
```

### Testing
```bash
# Run unit tests
pytest tests/

# Run specific test file
pytest tests/test_activations.py

# Run with coverage
pytest --cov=. tests/
```

### Code Quality
```bash
# Type checking
mypy .

# Linting
ruff check .

# Format code
ruff format .
```

## Architecture Overview

### Core Components

The project implements a complex-valued neural network framework with the following key innovations:

1. **Novel Holomorphic Activations**:
   - `holomorphic_elu`: Complex ELU that maintains holomorphicity without magnitude operations
   - `holomorphic_swish`: Smooth, gated activation using complex sigmoid

2. **Comparison Baselines**:
   - CReLU: Separate ReLU on real and imaginary parts
   - ModReLU: Magnitude-based ReLU (breaks holomorphicity)
   - Complex Tanh: Standard complex hyperbolic tangent

### Planned Project Structure

```
holomorphic-networks/
├── activations.py          # Holomorphic and baseline activation functions
├── layers.py               # Complex-valued linear and convolutional layers
├── models.py               # Network architectures (feedforward, CNN)
├── neural_ode.py           # Continuous-time neural ODE models using diffrax
├── analysis.py             # Magnitude drift analysis and metrics
├── experiments/
│   ├── synthetic.py        # Complex polynomial learning tasks
│   ├── oscillatory.py      # Dynamic system modeling
│   ├── spirals.py          # Complex spiral classification
│   └── complex_mnist.py    # Phase-encoded vision tasks
├── visualization.py        # Complex plane plotting and animations
├── tests/                  # Unit tests for holomorphicity and gradients
└── notebooks/              # Jupyter notebooks for exploration
```

### Key Technical Considerations

1. **Complex Differentiability**: All activation functions must maintain holomorphicity (Cauchy-Riemann equations) to ensure proper gradient flow through complex networks.

2. **Magnitude Drift**: Track and analyze how signal magnitude evolves through deep networks. The novel activations are designed to prevent exponential growth or decay.

3. **JAX Integration**: 
   - Use `jax.numpy` for all array operations
   - Apply `@jax.jit` for performance-critical functions
   - Leverage automatic differentiation with `jax.grad` for complex gradients

4. **Data Types**: Support both `complex64` and `complex128` dtypes throughout the implementation.

5. **Neural ODE**: Implement continuous dynamics using diffrax library for solving complex-valued ODEs.

## Implementation Notes

- Always verify holomorphicity of new activation functions using JAX's automatic differentiation
- Initialize complex weights using appropriate schemes (e.g., complex Glorot initialization)
- Track magnitude statistics (mean, std, max) at each layer during forward pass
- Ensure all experiments include comparisons across different activation functions
- Generate publication-quality visualizations showing complex plane mappings

## Testing Requirements

- Unit tests must verify:
  - Holomorphicity of activation functions
  - Correct gradient flow through complex layers
  - Magnitude preservation properties
  - Complex linear algebra operations
  - Neural ODE integration accuracy