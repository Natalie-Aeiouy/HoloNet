"""Complex-valued neural network models."""

from typing import List, Optional, Dict, Tuple, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from .training import TrainingConfig
import jax
import jax.numpy as jnp
from jax import Array, random
from functools import partial

from .activations import get_activation, ACTIVATIONS
from .layers import ComplexLinear, ComplexLayerNorm, create_mlp_params


class ComplexMLP:
    """Complex-valued Multi-Layer Perceptron."""
    
    def __init__(
        self,
        layer_sizes: List[int],
        activation: str = 'h_elu',
        final_activation: Optional[str] = None,
        use_bias: bool = True,
        use_layer_norm: bool = False,
        dropout_rate: float = 0.0,
        dtype=jnp.complex64
    ):
        """Initialize Complex MLP.
        
        Args:
            layer_sizes: List of layer dimensions [input, hidden1, ..., output]
            activation: Activation function name for hidden layers
            final_activation: Activation for output layer (None for linear)
            use_bias: Whether to use bias in linear layers
            use_layer_norm: Whether to use layer normalization
            dropout_rate: Dropout probability (0 for no dropout)
            dtype: Data type for computations
        """
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.final_activation = final_activation
        self.use_bias = use_bias
        self.use_layer_norm = use_layer_norm
        self.dropout_rate = dropout_rate
        self.dtype = dtype
        
        self.num_layers = len(layer_sizes) - 1
        self.activation_fn = get_activation(activation)
        self.final_activation_fn = (
            get_activation(final_activation) if final_activation else None
        )
    
    def init_params(self, key: random.PRNGKey) -> Dict:
        """Initialize all model parameters.
        
        Args:
            key: Random key for initialization
            
        Returns:
            Dictionary containing all model parameters
        """
        params = {
            'layers': [],
            'layer_norms': [] if self.use_layer_norm else None
        }
        
        for i in range(self.num_layers):
            key, subkey = random.split(key)
            
            # Initialize linear layer
            linear = ComplexLinear(
                self.layer_sizes[i],
                self.layer_sizes[i + 1],
                use_bias=self.use_bias,
                dtype=self.dtype
            )
            params['layers'].append(linear.init_params(subkey))
            
            # Initialize layer norm if needed (except for last layer)
            if self.use_layer_norm and i < self.num_layers - 1:
                key, subkey = random.split(key)
                ln = ComplexLayerNorm(
                    self.layer_sizes[i + 1],
                    dtype=self.dtype
                )
                params['layer_norms'].append(ln.init_params(subkey))
        
        return params
    
    @partial(jax.jit, static_argnums=(0, 3, 4))
    def forward(
        self,
        params: Dict,
        x: Array,
        training: bool = False,
        key: Optional[random.PRNGKey] = None
    ) -> Tuple[Array, Dict]:
        """Forward pass through the network.
        
        Args:
            params: Model parameters
            x: Input array of shape (batch, input_dim)
            training: Whether in training mode (affects dropout)
            key: Random key for dropout (required if dropout_rate > 0 and training)
            
        Returns:
            Tuple of (output, auxiliary_data)
            auxiliary_data contains magnitude statistics per layer
        """
        aux_data = {
            'magnitudes': [],
            'activations': []
        }
        
        # Process through layers
        for i in range(self.num_layers):
            # Linear transformation
            linear = ComplexLinear(
                self.layer_sizes[i],
                self.layer_sizes[i + 1],
                use_bias=self.use_bias,
                dtype=self.dtype
            )
            x = linear(params['layers'][i], x)
            
            # Layer normalization (except last layer)
            if self.use_layer_norm and i < self.num_layers - 1:
                ln = ComplexLayerNorm(
                    self.layer_sizes[i + 1],
                    dtype=self.dtype
                )
                x = ln(params['layer_norms'][i], x)
            
            # Activation (except last layer unless specified)
            if i < self.num_layers - 1:
                x = self.activation_fn(x)
            elif self.final_activation_fn is not None:
                x = self.final_activation_fn(x)
            
            # Dropout
            if self.dropout_rate > 0 and training and key is not None:
                key, subkey = random.split(key)
                keep_prob = 1 - self.dropout_rate
                mask = random.bernoulli(subkey, keep_prob, x.shape)
                x = x * mask / keep_prob
            
            # Track magnitude statistics
            magnitudes = jnp.abs(x)
            aux_data['magnitudes'].append({
                'mean': jnp.mean(magnitudes),
                'std': jnp.std(magnitudes),
                'max': jnp.max(magnitudes),
                'min': jnp.min(magnitudes)
            })
            aux_data['activations'].append(x)
        
        return x, aux_data
    
    def __call__(
        self,
        params: Dict,
        x: Array,
        training: bool = False,
        key: Optional[random.PRNGKey] = None
    ) -> Array:
        """Simplified forward pass returning only output.
        
        Args:
            params: Model parameters
            x: Input array
            training: Whether in training mode
            key: Random key for dropout
            
        Returns:
            Output array
        """
        output, _ = self.forward(params, x, training, key)
        return output
    
    def train(
        self,
        train_data: Tuple[Array, Array],
        val_data: Optional[Tuple[Array, Array]] = None,
        config: Optional['TrainingConfig'] = None,
        initial_params: Optional[Dict] = None,
        key: Optional[random.PRNGKey] = None,
        backprop_method: str = 'jax_autodiff'
    ) -> Tuple[Dict, Dict]:
        """Train the model using specified backpropagation method.
        
        Args:
            train_data: Tuple of (train_inputs, train_targets)
            val_data: Optional validation data
            config: Training configuration (uses defaults if None)
            initial_params: Initial parameters (if None, initialize randomly)
            key: Random key for initialization
            backprop_method: Backpropagation method to use
                - 'naive_real': Treat complex as 2D real
                - 'split': Split real/imaginary gradients
                - 'jax_autodiff': JAX automatic differentiation (default)
                - 'wirtinger': Wirtinger calculus
                
        Returns:
            Tuple of (final_params, training_history)
        """
        from .training import ComplexTrainer, TrainingConfig
        
        if config is None:
            config = TrainingConfig(backprop_method=backprop_method)
        else:
            config.backprop_method = backprop_method
        
        if key is None:
            key = random.PRNGKey(42)
        
        # Initialize parameters if not provided
        if initial_params is None:
            initial_params = self.init_params(key)
        
        # Create trainer and train
        trainer = ComplexTrainer(self, config)
        final_params = trainer.train(
            train_data, val_data, initial_params, key
        )
        
        return final_params, trainer.history


class ComplexResNet:
    """Complex-valued Residual Network."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_blocks: int = 3,
        activation: str = 'h_elu',
        use_bias: bool = True,
        residual_scale: float = 1.0,
        dtype=jnp.complex64
    ):
        """Initialize Complex ResNet.
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            num_blocks: Number of residual blocks
            activation: Activation function name
            use_bias: Whether to use bias in linear layers
            residual_scale: Scaling factor for residual connections
            dtype: Data type for computations
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_blocks = num_blocks
        self.activation = activation
        self.use_bias = use_bias
        self.residual_scale = residual_scale
        self.dtype = dtype
        
        self.activation_fn = get_activation(activation)
    
    def init_params(self, key: random.PRNGKey) -> Dict:
        """Initialize all model parameters.
        
        Args:
            key: Random key for initialization
            
        Returns:
            Dictionary containing all model parameters
        """
        params = {
            'input_projection': None,
            'blocks': [],
            'output_projection': None
        }
        
        # Input projection
        key, subkey = random.split(key)
        input_proj = ComplexLinear(
            self.input_dim,
            self.hidden_dim,
            use_bias=self.use_bias,
            dtype=self.dtype
        )
        params['input_projection'] = input_proj.init_params(subkey)
        
        # Residual blocks
        for _ in range(self.num_blocks):
            key, subkey1 = random.split(key)
            key, subkey2 = random.split(key)
            
            block = {
                'linear1': ComplexLinear(
                    self.hidden_dim,
                    self.hidden_dim,
                    use_bias=self.use_bias,
                    dtype=self.dtype
                ).init_params(subkey1),
                'linear2': ComplexLinear(
                    self.hidden_dim,
                    self.hidden_dim,
                    use_bias=self.use_bias,
                    dtype=self.dtype
                ).init_params(subkey2)
            }
            params['blocks'].append(block)
        
        # Output projection
        key, subkey = random.split(key)
        output_proj = ComplexLinear(
            self.hidden_dim,
            self.output_dim,
            use_bias=self.use_bias,
            dtype=self.dtype
        )
        params['output_projection'] = output_proj.init_params(subkey)
        
        return params
    
    def residual_block(
        self,
        block_params: Dict,
        x: Array
    ) -> Array:
        """Apply a residual block.
        
        Args:
            block_params: Parameters for this block
            x: Input array
            
        Returns:
            Output with residual connection
        """
        identity = x
        
        # First linear + activation
        linear1 = ComplexLinear(
            self.hidden_dim,
            self.hidden_dim,
            use_bias=self.use_bias,
            dtype=self.dtype
        )
        out = linear1(block_params['linear1'], x)
        out = self.activation_fn(out)
        
        # Second linear
        linear2 = ComplexLinear(
            self.hidden_dim,
            self.hidden_dim,
            use_bias=self.use_bias,
            dtype=self.dtype
        )
        out = linear2(block_params['linear2'], out)
        
        # Residual connection
        out = identity + self.residual_scale * out
        out = self.activation_fn(out)
        
        return out
    
    @partial(jax.jit, static_argnums=(0,))
    def forward(
        self,
        params: Dict,
        x: Array
    ) -> Tuple[Array, Dict]:
        """Forward pass through the network.
        
        Args:
            params: Model parameters
            x: Input array
            
        Returns:
            Tuple of (output, auxiliary_data)
        """
        aux_data = {
            'magnitudes': [],
            'block_outputs': []
        }
        
        # Input projection
        input_proj = ComplexLinear(
            self.input_dim,
            self.hidden_dim,
            use_bias=self.use_bias,
            dtype=self.dtype
        )
        x = input_proj(params['input_projection'], x)
        x = self.activation_fn(x)
        
        # Track initial magnitude
        aux_data['magnitudes'].append({
            'mean': jnp.mean(jnp.abs(x)),
            'std': jnp.std(jnp.abs(x)),
            'max': jnp.max(jnp.abs(x)),
            'min': jnp.min(jnp.abs(x))
        })
        
        # Residual blocks
        for i, block_params in enumerate(params['blocks']):
            x = self.residual_block(block_params, x)
            
            # Track magnitude after each block
            aux_data['magnitudes'].append({
                'mean': jnp.mean(jnp.abs(x)),
                'std': jnp.std(jnp.abs(x)),
                'max': jnp.max(jnp.abs(x)),
                'min': jnp.min(jnp.abs(x))
            })
            aux_data['block_outputs'].append(x)
        
        # Output projection
        output_proj = ComplexLinear(
            self.hidden_dim,
            self.output_dim,
            use_bias=self.use_bias,
            dtype=self.dtype
        )
        x = output_proj(params['output_projection'], x)
        
        return x, aux_data
    
    def __call__(self, params: Dict, x: Array) -> Array:
        """Simplified forward pass returning only output.
        
        Args:
            params: Model parameters
            x: Input array
            
        Returns:
            Output array
        """
        output, _ = self.forward(params, x)
        return output


def create_model(
    model_type: str,
    **kwargs
) -> Tuple[Callable, Callable]:
    """Factory function to create models.
    
    Args:
        model_type: Type of model ('mlp' or 'resnet')
        **kwargs: Model-specific arguments
        
    Returns:
        Tuple of (model_instance, init_fn)
    """
    if model_type == 'mlp':
        model = ComplexMLP(**kwargs)
    elif model_type == 'resnet':
        model = ComplexResNet(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model, model.init_params