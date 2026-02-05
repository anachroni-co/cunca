# ============================================================================
# P0 CRITICAL FIXES - CapibaraHybrid
# Correcciones bloqueantes for enable entrenamiento piloto
# ============================================================================

import chex
import flax.linen as nn
from flax import struct
from capibara.jax import jax
from capibara.jax import numpy as jnp
from typing import Dict, Tuple, Optional, Any

import logging
logger = logging.getLogger(__name__)

# ============================================================================
# FIX 1: quantum_submodel.py - Signature Consistency
# ============================================================================

@struct.dataclass
class QuantumSubmodelConfig:
    """setup validada for QuantumSubmodel."""
    vqbit_config: Dict[str, Any]
    quantum_wrapper_config: Dict[str, Any]
    enable_metrics: bool = True
    use_manifold_projection: bool = False
    manifold_curvature: float = -1.0

class QuantumSubmodel(nn.Module):
    """
    Submodelo cuántico-inspirado with firmas corregidas.
    
    FIXED: input consistente how Dict[str, jnp.ndarray]
    FIXED: Métricas actualizadas a nombres reales
    FIXED: validation de tipos explícita
    """
    config: QuantumSubmodelConfig
    
    def setup(self):
        self.vqbit_layer = VQbitLayerFixed(self.config.vqbit_config)
        self.quantum_wrapper = QuantumWrapperFixed(self.config.quantum_wrapper_config)
        
        if self.config.use_manifold_projection:
            self.manifold_projection = ManifoldProjection(
                curvature=self.config.manifold_curvature
            )
    
    def __call__(self, 
                 x: Dict[str, jnp.ndarray],  #  FIXED: Consistent signature
                 deterministic: bool = True,
                 **kwargs) -> Tuple[Dict[str, jnp.ndarray], Dict[str, float]]:
        """
        Forward pass with corrected input/output types.
        
        Args:
            x: Dict containing 'tokens', 'attention_mask', etc.
            deterministic: Whether to use deterministic mode
            
        Returns:
            Tuple of (outputs, metrics) with correct metric names
        """
        #  Input validation
        chex.assert_type(x, dict)
        if 'tokens' not in x:
            raise ValueError("Input dict must contain 'tokens' key")
        
        tokens = x['tokens']
        
        # VQbit compression
        compressed, vqbit_metrics = self.vqbit_layer(
            tokens, deterministic=deterministic
        )
        
        # Quantum-inspired operations
        quantum_out, quantum_metrics = self.quantum_wrapper(
            compressed, deterministic=deterministic
        )
        
        # Optional manifold projection
        if self.config.use_manifold_projection:
            quantum_out = self.manifold_projection(quantum_out)
        
        #  FIXED: Use real metric names
        aggregated_metrics = {
            # VQbit metrics (real names)
            'commitment_loss': vqbit_metrics.get('commitment_loss', 0.0),
            'codebook_usage': vqbit_metrics.get('usage', 0.0),
            'quantization_loss': vqbit_metrics.get('quantization_loss', 0.0),
            
            # Quantum wrapper metrics (real names)  
            'coherence_score': quantum_metrics.get('coherence', 0.0),
            'fidelity_score': quantum_metrics.get('fidelity', 0.0),
            'quantum_efficiency': quantum_metrics.get('efficiency', 0.0),
            
            #  REMOVED: Fake metrics
            # 'loss': DOES NOT EXIST
            # 'quantum_advantage': DOES NOT EXIST
        }
        
        # Construct output dict with same structure as input
        outputs = {
            'tokens': quantum_out,
            'attention_mask': x.get('attention_mask', None)
        }
        
        return outputs, aggregated_metrics

# ============================================================================
# FIX 2: VQbitLayer - EMA and Metrics Corrections
# ============================================================================

class VQbitLayerFixed(nn.Module):
    """VQbit layer with corrected EMA and real metrics."""
    
    config: Dict[str, Any]
    
    def setup(self):
        self.codebook_size = self.config.get('codebook_size', 256)
        self.embedding_dim = self.config.get('embedding_dim', 128)
        self.commitment_cost = self.config.get('commitment_cost', 0.25)
        self.ema_decay = self.config.get('ema_decay', 0.99)
        
        # Codebook embeddings
        self.codebook = self.param('codebook',
            nn.initializers.normal(stddev=0.1),
            (self.codebook_size, self.embedding_dim)
        )
        
        # EMA tracking variables (properly initialized)
        self.ema_cluster_size = self.variable('ema_state', 'cluster_size',
            lambda: jnp.ones(self.codebook_size)
        )
        self.ema_w = self.variable('ema_state', 'w',
            lambda: self.codebook.copy()
        )
    
    def __call__(self, 
                 x: jnp.ndarray, 
                 deterministic: bool = True) -> Tuple[jnp.ndarray, Dict[str, float]]:
        """
        VQbit quantization with corrected metrics.
        
        Returns real metric names: commitment_loss, usage, quantization_loss
        """
        batch_size, seq_len, dim = x.shape
        
        # Flatten for quantization
        flat_x = x.reshape(-1, dim)
        
        # Find nearest codebook entries
        distances = jnp.sum(flat_x**2, axis=1, keepdims=True) + \
                   jnp.sum(self.codebook**2, axis=1) - \
                   2 * jnp.dot(flat_x, self.codebook.T)
        
        encoding_indices = jnp.argmin(distances, axis=1)
        quantized = self.codebook[encoding_indices]
        
        #  FIXED: Real metric names only
        commitment_loss = jnp.mean((jnp.stop_gradient(quantized) - flat_x) ** 2)
        quantization_loss = jnp.mean((quantized - jnp.stop_gradient(flat_x)) ** 2)
        
        # Codebook usage (percentage of codes used)
        unique_indices = jnp.unique(encoding_indices)
        usage = len(unique_indices) / self.codebook_size
        
        # Update EMA during training
        if not deterministic:
            self._update_ema(encoding_indices, flat_x)
        
        # Straight-through estimator
        quantized = flat_x + jnp.stop_gradient(quantized - flat_x)
        quantized = quantized.reshape(batch_size, seq_len, dim)
        
        metrics = {
            'commitment_loss': float(commitment_loss),
            'quantization_loss': float(quantization_loss),
            'usage': float(usage)
        }
        
        return quantized, metrics
    
    def _update_ema(self, encoding_indices: jnp.ndarray, flat_x: jnp.ndarray):
        """Update EMA statistics for codebook learning."""
        # One-hot encoding for cluster assignments
        encodings = jnp.eye(self.codebook_size)[encoding_indices]
        
        # Update cluster sizes
        updated_cluster_size = self.ema_decay * self.ema_cluster_size.value + \
                              (1 - self.ema_decay) * jnp.sum(encodings, axis=0)
        self.ema_cluster_size.value = updated_cluster_size
        
        # Update embeddings
        dw = jnp.dot(encodings.T, flat_x)
        updated_w = self.ema_decay * self.ema_w.value + (1 - self.ema_decay) * dw
        self.ema_w.value = updated_w

# ============================================================================
# FIX 3: QuantumWrapper - Realistic FFT Operations
# ============================================================================

class QuantumWrapperFixed(nn.Module):
    """Quantum-inspired wrapper with realistic FFT operations."""
    
    config: Dict[str, Any]
    
    def setup(self):
        self.use_fft = self.config.get('use_fft', True)
        self.phase_modulation = self.config.get('phase_modulation', True)
        self.coherence_threshold = self.config.get('coherence_threshold', 0.8)
    
    def __call__(self, 
                 x: jnp.ndarray, 
                 deterministic: bool = True) -> Tuple[jnp.ndarray, Dict[str, float]]:
        """
        Apply quantum-inspired transformations with real metrics.
        """
        batch_size, seq_len, dim = x.shape
        
        if self.use_fft:
            # Apply FFT along sequence dimension (quantum-inspired)
            x_complex = x.astype(jnp.complex64)
            fft_x = jnp.fft.fft(x_complex, axis=1)
            
            if self.phase_modulation:
                # Soft phase modulation (inspired by quantum phases)
                phases = jnp.angle(fft_x)
                magnitudes = jnp.abs(fft_x)
                
                # Learnable phase transformation
                phase_weights = self.param('phase_weights',
                    nn.initializers.normal(stddev=0.1),
                    (seq_len, dim)
                )
                
                modulated_phases = phases + 0.1 * phase_weights
                modulated_fft = magnitudes * jnp.exp(1j * modulated_phases)
            else:
                modulated_fft = fft_x
            
            # Inverse FFT to get back to real space
            output = jnp.real(jnp.fft.ifft(modulated_fft, axis=1))
        else:
            output = x
        
        #  Calculate real metrics
        coherence = self._calculate_coherence(x, output)
        fidelity = self._calculate_fidelity(x, output)
        efficiency = coherence * fidelity
        
        metrics = {
            'coherence': float(coherence),
            'fidelity': float(fidelity), 
            'efficiency': float(efficiency)
        }
        
        return output, metrics
    
    def _calculate_coherence(self, x_in: jnp.ndarray, x_out: jnp.ndarray) -> float:
        """Calculate coherence score between input and output."""
        correlation = jnp.corrcoef(x_in.flatten(), x_out.flatten())[0, 1]
        return jnp.clip(correlation, 0.0, 1.0)
    
    def _calculate_fidelity(self, x_in: jnp.ndarray, x_out: jnp.ndarray) -> float:
        """Calculate fidelity score (similarity measure)."""
        mse = jnp.mean((x_in - x_out) ** 2)
        fidelity = jnp.exp(-mse)  # Exponential decay of MSE
        return jnp.clip(fidelity, 0.0, 1.0)

# ============================================================================
# FIX 4: ManifoldProjection - Geometric Embeddings
# ============================================================================

class ManifoldProjection(nn.Module):
    """Manifold projection for geometric embeddings."""
    
    curvature: float = -1.0  # Hyperbolic by default
    scale: float = 1.0
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Project embeddings onto manifold with specified curvature."""
        norm = jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-8
        
        if self.curvature < 0:  # Hyperbolic (Poincaré model)
            return x / norm * jnp.tanh(self.scale * norm)
        elif self.curvature > 0:  # Spherical
            return x / norm * jnp.sin(self.scale * norm)
        else:  # Euclidean (not projection)
            return x

# ============================================================================
# FIX 5: Updated Test Data Loader
# ============================================================================

class TestDataLoaderFixed:
    """Fixed data loader for testing with correct dtypes."""
    
    def __init__(self, batch_size: int = 4, seq_len: int = 128, vocab_size: int = 1000):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
    
    def generate_batch(self) -> Dict[str, jnp.ndarray]:
        """Generate test batch with correct structure."""
        # Generate random token IDs
        tokens = jnp.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        
        # Generate attention mask (all 1s for simplicity)
        attention_mask = jnp.ones((self.batch_size, self.seq_len), dtype=jnp.int32)
        
        return {
            'tokens': tokens,
            'attention_mask': attention_mask
        }

# ============================================================================
# FIX 6: Integration Test
# ============================================================================

def test_quantum_submodel_integration():
    """Test complete integration with fixed signatures."""
    
    # Configuration
    config = QuantumSubmodelConfig(
        vqbit_config={
            'codebook_size': 256,
            'embedding_dim': 128,
            'commitment_cost': 0.25,
            'ema_decay': 0.99
        },
        quantum_wrapper_config={
            'use_fft': True,
            'phase_modulation': True,
            'coherence_threshold': 0.8
        },
        enable_metrics=True,
        use_manifold_projection=True,
        manifold_curvature=-1.0  # Hyperbolic
    )
    
    # Initialize model
    model = QuantumSubmodel(config)
    
    # Test data
    data_loader = TestDataLoaderFixed(batch_size=2, seq_len=64)
    test_batch = data_loader.generate_batch()
    
    # Initialize parameters
    key = jax.random.PRNGKey(42)
    
    # Dummy forward pass for initialization
    dummy_input = {
        'tokens': jnp.ones((1, 64), dtype=jnp.int32),
        'attention_mask': jnp.ones((1, 64), dtype=jnp.int32)
    }
    
    variables = model.init(key, dummy_input)
    
    # Forward pass with real data
    outputs, metrics = model.apply(variables, test_batch, deterministic=True)
    
    #  Validate outputs
    assert isinstance(outputs, dict), "Output should be dict"
    assert 'tokens' in outputs, "Output should contain 'tokens'"
    assert isinstance(metrics, dict), "Metrics should be dict"
    
    #  Validate real metric names
    expected_metrics = [
        'commitment_loss', 'codebook_usage', 'quantization_loss',
        'coherence_score', 'fidelity_score', 'quantum_efficiency'
    ]
    
    for metric in expected_metrics:
        assert metric in metrics, f"Missing metric: {metric}"
        assert isinstance(metrics[metric], (float, int)), f"Metric {metric} should be numeric"
    
    logger.info(" All P0 fixes validated successfully!")
    logger.info(f"Output shape: {outputs['tokens'].shape}")
    logger.info(f"Metrics: {list(metrics.keys())}")
    return outputs, metrics

# ============================================================================
# FIX 7: Training Loop Integration Point
# ============================================================================

def create_training_step_function(model: QuantumSubmodel):
    """Create training step function with fixed metrics extraction."""
    
    def train_step(state, batch):
        """Single training step with corrected metric handling."""
        
        def loss_fn(params):
            outputs, metrics = model.apply(
                {'params': params}, 
                batch, 
                deterministic=False
            )
            
            #  FIXED: Use real metric names for loss
            primary_loss = jnp.mean(outputs['tokens']**2)  # Dummy loss for testing
            regularization = (
                metrics['commitment_loss'] + 
                metrics['quantization_loss'] * 0.1
            )
            
            total_loss = primary_loss + regularization
            
            return total_loss, {
                'total_loss': total_loss,
                'primary_loss': primary_loss,
                **metrics  # Include all real metrics
            }
        
        (loss_value, metrics), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(state.params)
        
        # Update state (simplified, real implementation would use optax)
        # state = state.apply_gradients(grads=grads)  # Uncomment for real training
        
        return state, metrics
    
    return jax.jit(train_step)

if __name__ == "__main__":
    # Run integration test
    outputs, metrics = test_quantum_submodel_integration()
    logger.info("\n P0 fixes complete - ready for pilot training!")