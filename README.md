# HoloNet: Holomorphic Complex-Valued Neural Networks

A JAX-based implementation of complex-valued neural networks with novel holomorphic activation functions that maintain complex differentiability while preventing magnitude drift in deep networks.

## Key Features

### Novel Holomorphic Activations
- **Holomorphic ELU (H-ELU)**: `f(z) = z if Re(z) > 0, else α(exp(z) - 1)`
- **Holomorphic Swish (H-Swish)**: `f(z) = z * sigmoid(βz)`

Both functions are fully complex-differentiable (holomorphic) and designed to preserve magnitude stability.

### Comprehensive Baseline Comparisons
- CReLU: `f(z) = ReLU(Re(z)) + i*ReLU(Im(z))`
- ModReLU: `f(z) = ReLU(|z| + b) * (z/|z|)`
- Complex Tanh, Sigmoid, zReLU, and more

### Advanced Analysis Tools
- Magnitude drift tracking through network layers
- Holomorphicity violation measurement
- Phase coherence analysis
- Spectral analysis of complex activations
- Publication-quality visualizations

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd HoloNet

# Create and activate virtual environment
python -m venv nvenv
source nvenv/bin/activate  # Linux/Mac
# or nvenv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Basic Usage

```python
import jax.numpy as jnp
from jax import random
from holomorphic_networks import ComplexMLP, holomorphic_elu, plot_complex_plane_mapping

# Create a complex-valued MLP
model = ComplexMLP(
    layer_sizes=[2, 64, 64, 1],
    activation='h_elu',  # Use holomorphic ELU
    dtype=jnp.complex64
)

# Initialize parameters
key = random.PRNGKey(42)
params = model.init_params(key)

# Forward pass with complex input
x = jnp.array([[1.0 + 1.0j], [0.5 - 0.8j]], dtype=jnp.complex64)
output, aux_data = model.forward(params, x)

# Analyze magnitude drift
from holomorphic_networks import analyze_magnitude_drift
metrics = analyze_magnitude_drift(
    lambda p, x: model.forward(p, x), 
    params, 
    x
)
print(f"Magnitude drift ratio: {metrics.drift_ratio:.3f}")

# Visualize activation function
plot_complex_plane_mapping(holomorphic_elu, save_path="h_elu_mapping.png")
```

### Running Experiments

```python
from holomorphic_networks.experiments.synthetic import compare_activations_polynomial

# Compare activation functions on polynomial learning
results = compare_activations_polynomial(
    polynomial_degree=2,
    activations=['h_elu', 'h_swish', 'crelu', 'tanh'],
    n_epochs=1000
)
```

## Architecture

### Core Components

```
holomorphic_networks/
├── activations.py          # All activation functions
├── layers.py               # Complex linear, conv layers  
├── models.py               # Network architectures
├── analysis.py             # Magnitude drift analysis
├── visualization.py        # Plotting utilities
├── experiments/
│   ├── synthetic.py        # Polynomial learning
│   ├── oscillatory.py      # Dynamic systems
│   ├── spirals.py          # Classification tasks
│   └── complex_mnist.py    # Vision tasks
└── tests/                  # Unit tests
```

### Key Technical Features

1. **Holomorphic Activations**: Maintain complex differentiability throughout the network
2. **Magnitude Preservation**: Novel activations prevent exponential magnitude growth/decay
3. **Comprehensive Analysis**: Track magnitude drift, phase coherence, and holomorphicity violations
4. **JAX Integration**: JIT compilation for high performance
5. **Complex Initialization**: Proper complex-valued weight initialization schemes

## Research Applications

### Synthetic Tasks
- Complex polynomial function learning (z², z³, etc.)
- Oscillatory dynamics modeling
- Complex spiral classification

### Real-World Applications  
- Phase-encoded signal processing
- Complex-valued computer vision
- Quantum state modeling
- Complex-valued time series prediction

## Development

### Running Tests
```bash
pytest tests/ -v
pytest tests/test_activations.py -v  # Test specific module
pytest --cov=holomorphic_networks tests/  # With coverage
```

### Code Quality
```bash
# Type checking
mypy holomorphic_networks/

# Linting and formatting  
ruff check holomorphic_networks/
ruff format holomorphic_networks/
```

### Development Workflow
1. Activate virtual environment: `source nvenv/bin/activate`
2. Make changes to code
3. Run tests: `pytest tests/`
4. Check types: `mypy .`
5. Format code: `ruff format .`

## Performance

The implementation uses JAX's JIT compilation for optimal performance:
- Automatic differentiation for complex gradients
- Vectorized operations for batch processing
- GPU/TPU acceleration support
- Memory-efficient complex operations

## Citation

If you use this code in your research, please cite:

```bibtex
@software{holonet2024,
  title={HoloNet: Holomorphic Complex-Valued Neural Networks},
  author={Natalie Aeiouy},
  year={2025},
  url={https://github.com/example/holonet}
}
```

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and add tests
4. Ensure tests pass: `pytest tests/`
5. Submit a pull request

## Related Work

- Trabelsi et al. "Deep Complex Networks" (2018)
- Guberman "On Complex Valued Convolutional Neural Networks" (2016)  
- Hirose & Yoshida "Generalization Characteristics of Complex-Valued Feedforward Neural Networks" (2012)

## Support

For questions or issues:
- Open an issue on GitHub
- Check the documentation in `CLAUDE.md`
- Review the test files for usage examples