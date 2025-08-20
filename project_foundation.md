#HoloNet
Project: Holomorphic Complex-Valued Neural Networks with Novel Activation Functions
I'm implementing a research project on complex-valued neural networks using JAX. The key innovation is two novel holomorphic activation functions that maintain complex differentiability while avoiding magnitude drift in deep networks.
Core Implementation Needed:

Create a JAX-based complex neural network framework with the following activation functions:

pythondef holomorphic_elu(z: Array, alpha: float = 1.0) -> Array:
    """
    Holomorphic Complex ELU activation function.
    
    For complex z = x + iy:
    f(z) = z if Re(z) > 0, else alpha * (exp(z) - 1)
    
    This is holomorphic everywhere and provides similar behavior to modrelu
    but without breaking holomorphicity by taking absolute values.
    """
    condition = z.real > 0
    return jnp.where(condition, z, alpha * (jnp.exp(z) - 1))

def holomorphic_swish(z: Array, beta: float = 1.0) -> Array:
    """
    Holomorphic Complex Swish activation: z * sigmoid(beta * z).
    
    f(z) = z * (1 / (1 + exp(-beta * z)))
    
    This is fully holomorphic and provides smooth, gated activation.
    """
    return z * jax.nn.sigmoid(beta * z)

Implement comparison baselines including:

CReLU: f(z) = ReLU(Re(z)) + i*ReLU(Im(z))
ModReLU: f(z) = ReLU(|z| + b) * (z/|z|) if |z| + b >= 0
Complex Tanh: f(z) = tanh(z)


Build a flexible complex-valued neural network class that:

Supports arbitrary depth and width
Uses complex64 or complex128 dtypes
Implements complex-valued linear layers with proper initialization
Supports both Neural ODE and standard feedforward architectures
Includes gradient computation using JAX's automatic differentiation


Create magnitude drift analysis tools:

Track magnitude statistics (mean, std, max) through each layer during forward pass
Visualize magnitude distributions at different depths
Compare magnitude preservation across different activation functions
Generate plots showing how magnitude evolves through 10, 50, and 100 layer networks


Implement benchmark experiments:

Synthetic task: Learn complex polynomial functions (z^2, z^3, etc.)
Oscillatory dynamics: Model sin/cos waves with phase shifts
Spiral classification: Complex version of the two-spirals problem
Complex MNIST: MNIST with phase-encoded features


Set up Neural ODE integration using diffrax:

Implement continuous-time complex neural ODE: dz/dt = f_θ(z(t), z*(t), t)
Compare discrete vs continuous dynamics with different activations
Visualize phase portraits in complex plane


Create comprehensive visualization suite:

Plot complex plane activations (input vs output mappings)
Animate how data flows through the network in complex space
Show gradient flow patterns for each activation
Generate publication-quality figures comparing all methods



Project Structure:
holomorphic-networks/
├── activations.py          # All activation functions
├── layers.py               # Complex linear, conv layers
├── models.py               # Network architectures
├── neural_ode.py           # ODE-based models
├── analysis.py             # Magnitude drift analysis
├── experiments/
│   ├── synthetic.py        # Polynomial learning
│   ├── oscillatory.py      # Dynamic systems
│   ├── spirals.py          # Classification tasks
│   └── complex_mnist.py    # Vision tasks
├── visualization.py        # Plotting utilities
├── tests/                  # Unit tests for all components
└── notebooks/              # Jupyter notebooks for exploration
Key Requirements:

Use JAX with JIT compilation for performance
Include type hints throughout
Add comprehensive docstrings
Implement unit tests for gradient flow and holomorphicity verification
Create a simple API for easy experimentation
Generate real-time training metrics showing magnitude statistics

Performance Tracking:
Track and compare:

Training/validation loss curves
Magnitude drift metrics per layer
Parameter count (showing efficiency gains)
Training time per epoch
Gradient norm statistics

Start by implementing the core activation functions and a simple feedforward network, then progressively add complexity. Make sure to verify that the H-ELU and H-Swish maintain proper complex gradients using JAX's automatic differentiation.
