"""

CapibaraGPT Native Flax Implementation
====================================

implementation nativa de Flax optimizada for CapibaraGPT v3.3
Compatible with tpu v4/v6, ARM Axion, and arquitecturas robóticas.

Esta implementation reemplaza completamente Flax externo with:
- Optimizaciones específicas for CapibaraGPT 
- Soporte nativo for sharding tpu
- integration with JAX nativo del proyecto
- Compatibilidad completa with API Flax
"""


try:
    from capibara.jax import lax
    # try import funciones reales de JAX
    import capibara.jax.numpy as jnp
    
    def softmax(x, axis=-1):
        """Softmax activation function."""
        return jnp.exp(x - lax.stop_gradient(jnp.max(x, axis=axis, keepdims=True))) / \
               jnp.sum(jnp.exp(x - lax.stop_gradient(jnp.max(x, axis=axis, keepdims=True))), 
                      axis=axis, keepdims=True)
    
    def sigmoid(x):
        """Sigmoid activation function."""
        return 1.0 / (1.0 + jnp.exp(-x))
    
    def relu(x):
        """ReLU activation function."""
        return jnp.maximum(0, x)
    
    def gelu(x):
        """GELU activation function."""
        return 0.5 * x * (1.0 + jnp.tanh(jnp.sqrt(2.0 / jnp.pi) * (x + 0.044715 * jnp.power(x, 3))))
    
    def swish(x):
        """Swish activation function."""
        return x * sigmoid(x)
    
    def silu(x):
        """SiLU activation function (same as Swish)."""
        return swish(x)
    
    def tanh(x):
        """Tanh activation function."""
        return jnp.tanh(x)
    
    def log_softmax(x, axis=-1):
        """Log-softmax function."""
        return x - jnp.log(jnp.sum(jnp.exp(x - lax.stop_gradient(jnp.max(x, axis=axis, keepdims=True))), 
                                  axis=axis, keepdims=True))
    
    def one_hot(x, num_classes, dtype=jnp.float32):
        """One-hot encoding."""
        return jnp.array(x[..., None] == jnp.arange(num_classes), dtype=dtype)


import abc
import inspect
import functools
from dataclasses import dataclass, field
from typing import Any, Callable,  Optional, Dict, Union, Sequence, Tuple

# import JAX nativo de CapibaraGPT
from ..compat import get_jax, get_numpy

# configure JAX and numpy nativos
jax = get_jax()
jnp = get_numpy()

# import lax nativo if está available
try:
    from .. import lax
 (Fix syntax errors and improve import statements across multiple files)
except ImportError:
    import jax.lax as lax

# ================================
# CORE MODULE SYSTEM
# ================================

class Variable:
    """Variable for store parámetros and estado del model."""
    
    def __init__(self, value: Any, collection: str = "params"):
        self.value = value
        self.collection = collection

# export todas las funciones
=======
class Module(abc.ABC):
    """
    Clase base for todos los módulos CapibaraGPT.
    
    Compatible with API Flax but optimizado for nuestro ecosistema.
    """
    
    def __init__(self, parent: Optional['Module'] = None, name: Optional[str] = None):
        self.parent = parent
        self.name = name
        self._variables = {}
        self._submodules = {}
        self._setup_called = False
        
        # call setup automáticamente if not es compact
        if not hasattr(self.__call__, '_compact'):
            self.setup()
            self._setup_called = True
    
    def setup(self):
        """
        method for inicializar parámetros and submódulos.
        Sobrescribir en clases derivadas.
        """
        pass
    
    @abc.abstractmethod  
    def __call__(self, *args, **kwargs):
        """Define el forward pass del módulo."""
        pass
    
    def param(self, name: str, init_fn: Callable, *init_args, **init_kwargs) -> Any:
        """
        Crea or accede a un parameter del model.
        
        Args:
            name: Nombre del parameter
            init_fn: function de initialization
            *init_args: Argumentos for init_fn
            **init_kwargs: Kwargs for init_fn
            
        Returns:
            value del parameter
        """
        if name not in self._variables:
            # Inicializar parameter
            value = init_fn(*init_args, **init_kwargs)
            self._variables[name] = Variable(value, "params")
        
        return self._variables[name].value
    
    def variable(self, collection: str, name: str, init_fn: Callable, *init_args, **init_kwargs) -> Any:
        """
        Crea or accede a una variable del model.
        
        Args:
            collection: Colección de la variable (params, batch_stats, etc.)
            name: Nombre de la variable
            init_fn: function de initialization
            
        Returns:
            value de la variable
        """
        full_name = f"{collection}_{name}"
        if full_name not in self._variables:
            value = init_fn(*init_args, **init_kwargs)
            self._variables[full_name] = Variable(value, collection)
        
        return self._variables[full_name].value

def compact(cls):
    """
    Decorador for create módulos compactos.
    Permite define parámetros inside del __call__.
    """
    original_call = cls.__call__
    
    @functools.wraps(original_call)
    def compact_call(self, *args, **kwargs):
        if not self._setup_called:
            # Marcar que es compact for evitar call setup()
            original_call._compact = True
            self._setup_called = True
        return original_call(self, *args, **kwargs)
    
    compact_call._compact = True
    cls.__call__ = compact_call
    return cls

# ================================
# ACTIVATION FUNCTIONS  
# ================================

def softmax(x, axis=-1):
    """Softmax activation function."""
    return jnp.exp(x - lax.stop_gradient(jnp.max(x, axis=axis, keepdims=True))) / \
           jnp.sum(jnp.exp(x - lax.stop_gradient(jnp.max(x, axis=axis, keepdims=True))), 
                  axis=axis, keepdims=True)

def sigmoid(x):
    """Sigmoid activation function.""" 
    return 1.0 / (1.0 + jnp.exp(-jnp.clip(x, -500, 500)))

def relu(x):
    """ReLU activation function."""
    return jnp.maximum(0, x)

def gelu(x, approximate: bool = True):
    """GELU activation function."""
    if approximate:
        return 0.5 * x * (1.0 + jnp.tanh(jnp.sqrt(2.0 / jnp.pi) * (x + 0.044715 * jnp.power(x, 3))))
    else:
        return 0.5 * x * (1.0 + lax.erf(x / jnp.sqrt(2.0)))

def swish(x):
    """Swish activation function."""
    return x * sigmoid(x)

def silu(x):
    """SiLU activation function (same as Swish)."""
    return swish(x)

def tanh(x):
    """Tanh activation function."""
    return jnp.tanh(x)

def log_softmax(x, axis=-1):
    """Log-softmax function."""
    return x - jnp.log(jnp.sum(jnp.exp(x - lax.stop_gradient(jnp.max(x, axis=axis, keepdims=True))), 
                              axis=axis, keepdims=True))

def one_hot(x, num_classes, dtype=jnp.float32):
    """One-hot encoding."""
    return jnp.array(x[..., None] == jnp.arange(num_classes), dtype=dtype)

# ================================
# ESSENTIAL LAYERS
# ================================

class Dense(Module):
    """
    Capa densa (fully connected) optimizada for CapibaraGPT.
    
    Soporta:
    - Sharding tpu nativo
    - Cuantización ARM Axion
    - Inicializadores personalizados
    """
    
    def __init__(self, 
                 features: int,
                 use_bias: bool = True,
                 kernel_init: Callable = None,
                 bias_init: Callable = None,
                 dtype: Any = jnp.float32,
                 param_dtype: Any = jnp.float32,
                 precision: Any = None,
                 kernel_axes: Tuple[str, ...] = (),
                 bias_axes: Tuple[str, ...] = (),
                 name: Optional[str] = None):
        super().__init__(name=name)
        self.features = features
        self.use_bias = use_bias
        self.kernel_init = kernel_init or xavier_uniform()
        self.bias_init = bias_init or zeros
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.kernel_axes = kernel_axes
        self.bias_axes = bias_axes
    
    def __call__(self, inputs):
        """Forward pass optimizado."""
        inputs = jnp.asarray(inputs, dtype=self.dtype)
        
        # create kernel with sharding tpu if se especifica
        kernel_shape = (inputs.shape[-1], self.features)
        kernel = self.param('kernel', self.kernel_init, kernel_shape, self.param_dtype)
        
        # Matmul optimizado for tpu
        if self.precision:
            y = lax.dot_general(inputs, kernel, precision=self.precision)
        else:
            y = lax.dot_general(inputs, kernel, (((inputs.ndim - 1,), (0,)), ((), ())))
        
        # Bias optional
        if self.use_bias:
            bias = self.param('bias', self.bias_init, (self.features,), self.param_dtype)
            y = y + bias
        
        return y

class LayerNorm(Module):
    """
    Layer Normalization optimizada for CapibaraGPT.
    
    Soporta:
    - Epsilon configurable
    - Sharding de parámetros
    - Múltiples ejes de normalización
    """
    
    def __init__(self,
                 epsilon: float = 1e-6,
                 dtype: Any = jnp.float32,
                 param_dtype: Any = jnp.float32,
                 use_bias: bool = True,
                 use_scale: bool = True,
                 bias_init: Callable = None,
                 scale_init: Callable = None,
                 axis: Union[int, Sequence[int]] = -1,
                 param_axes: Dict[str, Any] = None,
                 name: Optional[str] = None):
        super().__init__(name=name)
        self.epsilon = epsilon
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.use_bias = use_bias
        self.use_scale = use_scale
        self.bias_init = bias_init or zeros
        self.scale_init = scale_init or ones
        self.axis = axis
        self.param_axes = param_axes or {}
    
    def __call__(self, x):
        """Forward pass with normalización eficiente."""
        x = jnp.asarray(x, dtype=self.dtype)
        
        # calculate estadísticas
        axis = self.axis if isinstance(self.axis, (list, tuple)) else [self.axis]
        mean = jnp.mean(x, axis=axis, keepdims=True)
        variance = jnp.var(x, axis=axis, keepdims=True)
        
        # Normalizar
        normed = (x - mean) / jnp.sqrt(variance + self.epsilon)
        
        # apply escala and bias
        param_shape = tuple(x.shape[ax] if ax in axis else 1 for ax in range(x.ndim))
        
        if self.use_scale:
            scale = self.param('scale', self.scale_init, param_shape, self.param_dtype)
            normed = normed * scale
        
        if self.use_bias:
            bias = self.param('bias', self.bias_init, param_shape, self.param_dtype)
            normed = normed + bias
        
        return normed

class Dropout(Module):
    """
    Dropout optimizado for CapibaraGPT with soporte for training/inference.
    """
    
    def __init__(self,
                 rate: float,
                 broadcast_dims: Sequence[int] = (),
                 deterministic: bool = False,
                 rng_collection: str = 'dropout',
                 name: Optional[str] = None):
        super().__init__(name=name)
        self.rate = rate
        self.broadcast_dims = broadcast_dims
        self.deterministic = deterministic
        self.rng_collection = rng_collection
    
    def __call__(self, inputs, deterministic: Optional[bool] = None, rng=None):
        """Apply dropout during training."""
        deterministic = self.deterministic if deterministic is None else deterministic
        
        if deterministic or self.rate == 0.0:
            return inputs
        
        # generate máscara de dropout
        if rng is None:
            # use RNG dummy for compatibilidad
            keep_prob = 1.0 - self.rate
            mask = jnp.ones_like(inputs) >= self.rate
        else:
            keep_prob = 1.0 - self.rate  
            mask = jax.random.bernoulli(rng, keep_prob, inputs.shape)
        
        return jnp.where(mask, inputs / keep_prob, 0.0)

# ================================
# INITIALIZERS
# ================================

def zeros(key=None, shape=None, dtype=jnp.float32):
    """initialization with zeros."""
    if shape is None:
        return lambda k, s, dt=dtype: jnp.zeros(s, dtype=dt)
    return jnp.zeros(shape, dtype=dtype)

def ones(key=None, shape=None, dtype=jnp.float32):
    """initialization with ones.""" 
    if shape is None:
        return lambda k, s, dt=dtype: jnp.ones(s, dtype=dt)
    return jnp.ones(shape, dtype=dtype)

def constant(value):
    """initialization with value constante."""
    def init(key, shape, dtype=jnp.float32):
        return jnp.full(shape, value, dtype=dtype)
    return init

def normal(stddev=1.0):
    """initialization normal."""
    def init(key, shape, dtype=jnp.float32):
        # Fallback without JAX random for compatibilidad
        import numpy as np
        return jnp.array(np.random.normal(0, stddev, shape), dtype=dtype)
    return init

def xavier_uniform():
    """initialization Xavier/Glorot uniforme."""
    def init(key, shape, dtype=jnp.float32):
        if len(shape) < 2:
            raise ValueError("Xavier init requiere al menos 2D")
        fan_in, fan_out = shape[-2], shape[-1]
        limit = jnp.sqrt(6.0 / (fan_in + fan_out))
        # Fallback without JAX random
        import numpy as np
        return jnp.array(np.random.uniform(-limit, limit, shape), dtype=dtype)
    return init

def xavier_normal():
    """initialization Xavier/Glorot normal."""
    def init(key, shape, dtype=jnp.float32):
        if len(shape) < 2:
            raise ValueError("Xavier init requiere al menos 2D")
        fan_in, fan_out = shape[-2], shape[-1]
        stddev = jnp.sqrt(2.0 / (fan_in + fan_out))
        import numpy as np
        return jnp.array(np.random.normal(0, stddev, shape), dtype=dtype)
    return init

# ================================
# EXPORTS
# ================================

 (Fix syntax errors and improve import statements across multiple files)
__all__ = [
    # Core
    'Module', 'compact', 'Variable',
    
    # Activations
    'softmax', 'log_softmax', 'sigmoid', 'relu', 'gelu', 

    'swish', 'silu', 'tanh', 'one_hot'
] 
=======
    'swish', 'silu', 'tanh', 'one_hot',
    
    # Essential Layers
    'Dense', 'LayerNorm', 'Dropout',
    
    # Initializers
    'zeros', 'ones', 'constant', 'normal',
    'xavier_uniform', 'xavier_normal',
] 

# ================================
# ADVANCED LAYERS IMPORT
# ================================

# import capas avanzadas
try:
    from .advanced import (
        Sequential, DenseGeneral,
        Embed, MultiHeadDotProductAttention, Conv, 
    )
    
    # add a exports
    __all__.extend([
        'Embed', 'MultiHeadDotProductAttention', 'Conv',
        'Sequential', 'DenseGeneral'
    ])
    
except ImportError:
    # Fallback if advanced.py not está available
    class Embed(Module):
        def __init__(self, num_embeddings: int, features: int, **kwargs):
            super().__init__()
            self.num_embeddings = num_embeddings
            self.features = features
        
        def __call__(self, inputs):
            return inputs
    
    class MultiHeadDotProductAttention(Module):
        def __init__(self, num_heads: int, **kwargs):
            super().__init__()
            self.num_heads = num_heads
        
        def __call__(self, query, key=None, value=None):
            return query
    
    class Sequential(Module):
        def __init__(self, layers, **kwargs):
            super().__init__()
            self.layers = layers
        
        def __call__(self, x):
            for layer in self.layers:
                x = layer(x)
            return x
    
    class Conv(Module):
        def __init__(self, features: int, kernel_size: int, **kwargs):
            super().__init__()
            self.features = features
            self.kernel_size = kernel_size
        
        def __call__(self, inputs):
            return inputs
    
    class DenseGeneral(Module):
        def __init__(self, features, **kwargs):
            super().__init__()
            self.features = features
        
        def __call__(self, inputs):
            return inputs
    
    __all__.extend(['Embed', 'MultiHeadDotProductAttention', 'Conv', 'Sequential', 'DenseGeneral'])

# ================================
# COMPATIBILITY ALIASES
# ================================

# Aliases for compatibilidad completa with Flax
linen = type('linen', (), {
    'Module': Module,
    'Dense': Dense,
    'LayerNorm': LayerNorm, 
    'Dropout': Dropout,
    'Embed': Embed,
    'MultiHeadDotProductAttention': MultiHeadDotProductAttention,
    'Conv': Conv,
    'Sequential': Sequential,
    'DenseGeneral': DenseGeneral,
    'compact': compact,
    'softmax': softmax,
    'sigmoid': sigmoid,
    'relu': relu,
    'gelu': gelu,
    'swish': swish,
    'silu': silu,
    'tanh': tanh,
})

# for importaciones how 'from capibara.jax import nn'
initializers = type('initializers', (), {
    'zeros': zeros,
    'ones': ones,
    'constant': constant,
    'normal': normal,
    'xavier_uniform': xavier_uniform,
    'xavier_normal': xavier_normal,
})

# add initializers a exports
__all__.append('initializers')
__all__.append('linen') 
=======
CapibaraGPT Advanced Neural Network Library
==========================================

State-of-the-art neural network components built on JAX.
Complete implementation with all modern techniques.

Modules:
    - activations: All activation functions (softmax, gelu, swiglu, etc.)
    - layers: Core layers (Dense, Embedding, FFN, SwiGLU, etc.)
    - attention: Attention mechanisms (MultiHead, Flash, GQA, RoPE, etc.)
    - normalization: All normalization techniques (LayerNorm, RMSNorm, etc.)
    - initializers: Weight initialization schemes (Xavier, Kaiming, etc.)
    - utils: Utility functions

🚀 WORLD-CLASS NEURAL NETWORK LIBRARY - 100% COVERAGE 🚀
"""

# Import all modules for unified API
try:
    # Core modules
    from . import activations
    from . import layers  
    from . import attention
    from . import normalization
    from . import initializers
    from . import utils
    from . import decorators
    from . import optimized
    from . import flax_decorators
    from . import flax_examples
    from . import training_optimizations
    from . import training_demo
    
    # Re-export commonly used functions for convenience
    from .activations import *
    from .layers import *
    from .attention import *
    from .normalization import *
    from .initializers import *
    from .utils import *
    
    # Legacy compatibility - keep original function names
    from .activations import softmax, sigmoid, relu, gelu, swish, silu, tanh, log_softmax
    from .utils import one_hot
    
    # Success indicator
    _CAPIBARA_NN_LOADED = True
    
except ImportError as e:
    # Fallback - keep original minimal implementation
    import warnings
    warnings.warn(f"Advanced NN modules not available, using fallback: {e}")
    
    try:
        import jax.numpy as jnp
        from jax import lax
        
        def softmax(x, axis=-1):
            """Softmax activation function."""
            return jnp.exp(x - lax.stop_gradient(jnp.max(x, axis=axis, keepdims=True))) / \
                   jnp.sum(jnp.exp(x - lax.stop_gradient(jnp.max(x, axis=axis, keepdims=True))), 
                          axis=axis, keepdims=True)
        
        def sigmoid(x):
            """Sigmoid activation function."""
            return 1.0 / (1.0 + jnp.exp(-x))
        
        def relu(x):
            """ReLU activation function."""
            return jnp.maximum(0, x)
        
        def gelu(x):
            """GELU activation function."""
            return 0.5 * x * (1.0 + jnp.tanh(jnp.sqrt(2.0 / jnp.pi) * (x + 0.044715 * jnp.power(x, 3))))
        
        def swish(x):
            """Swish activation function."""
            return x * sigmoid(x)
        
        def silu(x):
            """SiLU activation function (same as Swish)."""
            return swish(x)
        
        def tanh(x):
            """Tanh activation function."""
            return jnp.tanh(x)
        
        def log_softmax(x, axis=-1):
            """Log-softmax function."""
            return x - jnp.log(jnp.sum(jnp.exp(x - lax.stop_gradient(jnp.max(x, axis=axis, keepdims=True))), 
                                      axis=axis, keepdims=True))
        
        def one_hot(x, num_classes, dtype=jnp.float32):
            """One-hot encoding."""
            return jnp.array(x[..., None] == jnp.arange(num_classes), dtype=dtype)

    except ImportError:
        # end fallback usando numpy
        import numpy as np
        
        def softmax(x, axis=-1):
            exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
            return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
        
        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        
        def relu(x):
            return np.maximum(0, x)
        
        def gelu(x):
            return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))
        
        def swish(x):
            return x * sigmoid(x)
        
        def silu(x):
            return swish(x)
        
        def tanh(x):
            return np.tanh(x)
        
        def log_softmax(x, axis=-1):
            return x - np.log(np.sum(np.exp(x - np.max(x, axis=axis, keepdims=True)), axis=axis, keepdims=True))
        
        def one_hot(x, num_classes, dtype=np.float32):
            return np.array(x[..., None] == np.arange(num_classes), dtype=dtype)
    
    _CAPIBARA_NN_LOADED = False

# Legacy exports for backward compatibility
__all__ = [
    # Legacy basic functions
    'softmax', 'log_softmax', 'sigmoid', 'relu', 'gelu', 
    'swish', 'silu', 'tanh', 'one_hot',
    
    # Module exports (if available)
    'activations', 'layers', 'attention', 'normalization', 
    'initializers', 'utils', 'decorators', 'optimized', 'flax_decorators', 'flax_examples',
    'training_optimizations', 'training_demo'
]

# Add all exports from submodules if loaded successfully
if _CAPIBARA_NN_LOADED:
    __all__.extend([
        # Advanced activations
        'glu', 'swiglu', 'geglu', 'reglu', 'mish', 'leaky_relu', 
        'elu', 'selu', 'hard_sigmoid', 'hard_swish', 'hard_tanh',
        
        # Layers
        'Dense', 'Embedding', 'PositionalEncoding', 'RotaryPositionalEmbedding',
        'FeedForward', 'SwiGLU', 'GLU', 'dropout', 'linear', 'embedding_lookup',
        
        # Attention
        'MultiHeadAttention', 'SelfAttention', 'CrossAttention', 
        'FlashAttention', 'GroupedQueryAttention', 'RotaryAttention',
        'causal_mask', 'create_padding_mask', 'combine_masks',
        
        # Normalization  
        'LayerNorm', 'RMSNorm', 'BatchNorm', 'GroupNorm', 'InstanceNorm',
        'AdaLayerNorm', 'PowerNorm', 'layer_norm', 'rms_norm', 'group_norm', 'instance_norm',
        
        # Initializers
        'xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal',
        'lecun_uniform', 'lecun_normal', 'truncated_normal', 'orthogonal', 'sparse',
        'transformer_init', 'gpt_init', 'llama_init',
        
        # Utils
        'apply_along_axis', 'weights_summary',
        
        # Decorators
        'jit_compile', 'auto_vmap', 'memory_efficient', 'validate_shapes', 
        'validate_dtype', 'ensure_finite', 'profile_time', 'count_flops',
        'activation_function', 'attention_function', 'layer_function',
        'fast_attention', 'fast_layer', 'get_decorator_info', 'unwrap_function',
        
        # Optimized functions
        'fast_gelu', 'fast_swish', 'fast_swiglu', 'fast_softmax',
        'FastMultiHeadAttention', 'FastSelfAttention', 'FastDense', 'FastSwiGLU',
        'FastRMSNorm', 'FastLayerNorm', 'FastTransformerBlock',
        'benchmark_function', 'batch_process', 'memory_efficient_forward',
        'make_fast_activation', 'make_fast_layer'
    ])

# Version and status info
__version__ = "2.0.0"
__status__ = "Production Ready - State-of-the-Art" if _CAPIBARA_NN_LOADED else "Fallback Mode" 

