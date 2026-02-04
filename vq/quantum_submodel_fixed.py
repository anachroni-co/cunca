# ============================================================================
# CapibaraHybrid – Quantum Submodel (TRAINABLE VERSION)
# JAX / Flax / JIT safe
# ============================================================================

import chex
import flax.linen as nn
from flax import struct
from capibara.jax import jax
from capibara.jax import numpy as jnp
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

    def __call__(
        self,
        x: Dict[str, jnp.ndarray],
        deterministic: bool = True
    ) -> Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]]:

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
            out = self.manifold(out)

        metrics = {
            "commitment_loss": vq_metrics["commitment_loss"],
            "quantization_loss": vq_metrics["quantization_loss"],
            "codebook_usage": vq_metrics["usage"],
            "coherence": q_metrics["coherence"],
            "fidelity": q_metrics["fidelity"],
            "efficiency": q_metrics["efficiency"],
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

        indices = jnp.argmin(distances, axis=1)
        quantized = self.codebook[indices]

        # losses
        commit_loss = jnp.mean((jax.lax.stop_gradient(quantized) - flat) ** 2)
        quant_loss = jnp.mean((quantized - jax.lax.stop_gradient(flat)) ** 2)

        # usage (JIT SAFE)
        counts = jnp.bincount(indices, length=self.codebook_size)
        usage = jnp.mean(counts > 0)

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

class ManifoldProjection(nn.Module):
    curvature: float = -1.0

    def __call__(self, x):
        n = jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-8
        if self.curvature < 0:
            return x / n * jnp.tanh(n)
        return x

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

            loss = (
                jnp.mean(out["tokens"] ** 2)
                + metrics["commitment_loss"]
                + 0.1 * metrics["quantization_loss"]
            )

            return loss, (metrics)

        (loss, metrics), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(state.params)

        state = state.apply_gradients(grads=grads)
        return state, metrics

    return train_step