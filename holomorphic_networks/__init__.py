"""Holomorphic Neural Networks: Complex-valued neural networks with novel activation functions."""

from .activations import (
    holomorphic_elu,
    holomorphic_swish,
    crelu,
    modrelu,
    complex_tanh,
    get_activation,
    ACTIVATIONS
)

from .layers import (
    ComplexLinear,
    ComplexConv2D,
    ComplexLayerNorm,
    ComplexBatchNorm,
    complex_glorot_normal,
    complex_glorot_uniform,
    complex_he_normal,
    complex_he_uniform
)

from .models import (
    ComplexMLP,
    ComplexResNet,
    create_model
)

from .analysis import (
    analyze_magnitude_drift,
    track_magnitude_evolution,
    compute_holomorphicity_violation,
    comprehensive_analysis,
    MagnitudeStats,
    MagnitudeDriftMetrics
)

from .visualization import (
    plot_complex_plane_mapping,
    plot_magnitude_drift,
    plot_activation_comparison,
    create_training_dashboard,
    PlotConfig
)

__version__ = "0.1.0"
__author__ = "HoloNet Team"
__description__ = "Holomorphic complex-valued neural networks with JAX"