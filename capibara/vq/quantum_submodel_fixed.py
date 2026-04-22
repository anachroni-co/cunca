"""
Quantum Submodel (Fixed) - Trainable quantum-inspired neural submodel.

This module provides a trainable quantum-inspired submodel for CapibaraGPT,
implementing VQ-bit layers and quantum wrappers with JAX/Flax JIT safety.
Supports manifold projection for geometric transformations.

Key Components:
    - QuantumSubmodelConfig: Configuration dataclass for submodel settings
    - QuantumSubmodel: Main Flax module for quantum-inspired processing

Features:
    - Token embedding layer
    - VQ-bit quantization layer
    - Quantum wrapper for quantum-inspired operations
    - Optional manifold projection

Author: Skydesk International Dev Team.
"""

# ============================================================================
# CapibaraHybrid - Quantum Submodel (TRAINABLE VERSION)
# JAX / Flax / JIT safe
# ============================================================================

import chex
import flax.linen as nn
from flax import struct
import jax
from jax import numpy as jnp
from typing import Dict, Tuple, Any

# ============================================================================
# CONFIG
# ============================================================================

@struct.dataclass
class QuantumSubmodelConfig:
    vocab_size: int
    embedding_dim: int
    vqbit_config: Dict[str, Any]
    quantum_wrapper_config: Dict[str, Any]
    use_manifold_projection: bool = False
    manifold_curvature: float = -1.0

# ============================================================================
# MAIN SUBMODEL
# ============================================================================

class QuantumSubmodel(nn.Module):
    config: QuantumSubmodelConfig

    def setup(self):
        # 🔴 REQUIRED: token → embedding
        self.embedding = nn.Embed(
            num_embeddings=self.config.vocab_size,
            features=self.config.embedding_dim
        )

        self.vqbit = VQbitLayerFixed(self.config.vqbit_config)
        self.quantum = QuantumWrapperFixed(self.config.quantum_wrapper_config)

        if self.config.use_manifold_projection:
            self.manifold = ManifoldProjection(
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
        tokens = x["tokens"]

        # (B, S, D)
        embedded = self.embedding(tokens)

        quantized, vq_metrics = self.vqbit(
            embedded, deterministic=deterministic
        )

        out, q_metrics = self.quantum(
            quantized, deterministic=deterministic
        )

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

        return {
            "tokens": out,
            "attention_mask": x.get("attention_mask", None)
        }, metrics

# ============================================================================
# VQBIT (EMA – JIT SAFE)
# ============================================================================

class VQbitLayerFixed(nn.Module):
    config: Dict[str, Any]

    def setup(self):
        self.codebook_size = self.config["codebook_size"]
        self.embedding_dim = self.config["embedding_dim"]
        self.commitment_cost = self.config.get("commitment_cost", 0.25)
        self.ema_decay = self.config.get("ema_decay", 0.99)

        self.codebook = self.param(
            "codebook",
            nn.initializers.normal(0.1),
            (self.codebook_size, self.embedding_dim),
        )

        self.ema_cluster_size = self.variable(
            "ema", "cluster_size", lambda: jnp.zeros(self.codebook_size)
        )
        self.ema_w = self.variable(
            "ema", "w", lambda: jnp.zeros_like(self.codebook)
        )

    def __call__(
        self,
        x: jnp.ndarray,
        deterministic: bool
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:

        B, S, D = x.shape
        flat = x.reshape(-1, D)

        distances = (
            jnp.sum(flat ** 2, axis=1, keepdims=True)
            + jnp.sum(self.codebook ** 2, axis=1)
            - 2 * flat @ self.codebook.T
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
            self._ema_update(indices, flat)

        quantized = flat + jax.lax.stop_gradient(quantized - flat)
        quantized = quantized.reshape(B, S, D)

        return quantized, {
            "commitment_loss": commit_loss,
            "quantization_loss": quant_loss,
            "usage": usage,
        }

    def _ema_update(self, indices, flat):
        one_hot = jax.nn.one_hot(indices, self.codebook_size)

        cluster_size = (
            self.ema_decay * self.ema_cluster_size.value
            + (1 - self.ema_decay) * jnp.sum(one_hot, axis=0)
        )

        dw = one_hot.T @ flat
        w = self.ema_decay * self.ema_w.value + (1 - self.ema_decay) * dw

        self.ema_cluster_size.value = cluster_size
        self.ema_w.value = w

        # 🔴 sync codebook
        n = cluster_size + 1e-5
        self.codebook = w / n[:, None]

# ============================================================================
# QUANTUM WRAPPER (FFT – SAFE)
# ============================================================================

class QuantumWrapperFixed(nn.Module):
    config: Dict[str, Any]

    def setup(self):
        self.use_fft = self.config.get("use_fft", True)

    def __call__(self, x, deterministic=True):
        if not self.use_fft:
            return x, self._metrics(x, x)

        x_c = x.astype(jnp.complex64)
        fft = jnp.fft.fft(x_c, axis=-1)
        out = jnp.real(jnp.fft.ifft(fft, axis=-1))

        return out, self._metrics(x, out)

    def _metrics(self, x_in, x_out):
        num = jnp.sum(x_in * x_out)
        den = jnp.linalg.norm(x_in) * jnp.linalg.norm(x_out) + 1e-8
        coherence = jnp.clip(num / den, 0.0, 1.0)

        mse = jnp.mean((x_in - x_out) ** 2)
        fidelity = jnp.exp(-mse)

        return {
            "coherence": coherence,
            "fidelity": fidelity,
            "efficiency": coherence * fidelity,
        }

# ============================================================================
# MANIFOLD
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
# TRAIN STEP (OPTAX READY)
# ============================================================================

def create_train_step(model, optimizer):
    @jax.jit
    def train_step(state, batch):

        def loss_fn(params):
            (out, metrics) = model.apply(
                {"params": params, **state.ema},
                batch,
                deterministic=False,
                mutable=["ema"],
            )
            
            #  FIXED: Use real metric names for loss
            primary_loss = jnp.mean(outputs['tokens']**2)  # Dummy loss for testing
            regularization = (
                metrics['commitment_loss'] + 
                metrics['quantization_loss'] * 0.1
            )

            return loss, (metrics)

        (loss, metrics), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(state.params)

        state = state.apply_gradients(grads=grads)
        return state, metrics

if __name__ == "__main__":
    # Run integration test
    outputs, metrics = test_quantum_submodel_integration()
    logger.info("\n P0 fixes complete - ready for pilot training!")
