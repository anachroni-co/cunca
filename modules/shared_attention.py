"""
Shared Attention Module - Optimizado for tpu v4-32
Atención multi-cabeza eficiente with fusión de operaciones.
"""

import os
import sys
import math
# Obtiene la path del directory current (scripts) -> /.../scripts
script_dir = os.path.dirname(os.path.abspath(__file__))
# Sube un level for obtain la raíz del proyecto -> /.../capibaraGPT-v2
project_root = os.path.dirname(script_dir)
# Añade la raíz del proyecto a sys.path
if project_root not in sys.path:
    sys.path.append(project_root)

from capibara.jax import jax
from flax import linen as nn
from functools import partial
from capibara.jax import numpy as jnp
from typing import Optional, Dict, Any, Tuple
from capibara.modules.shared_attention import create_shared_attention

class OptimizedSharedAttention(nn.Module):
    """
    Atención compartida optimizada for tpu v4-32.
    
    Características:
    - Fused QKV computation
    - Memory-efficient attention
    - Optimized for BF16
    - Minimal memory footprint
    """
    
    hidden_size: int
    num_heads: int
    dropout_rate: float = 0.1
    dtype: Any = jnp.bfloat16
    param_dtype: Any = jnp.float32
    use_bias: bool = False  # more efficient in tpu
    
    def setup(self):
        """initialization optimizada."""
        
        assert self.hidden_size % self.num_heads == 0, "hidden_size debe ser divisible por num_heads"
        
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # 1. Proyección QKV fusionada (more eficiente que 3 separadas)
        self.qkv_projection = nn.Dense(
            features=3 * self.hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=self.use_bias,
            kernel_init=nn.initializers.lecun_normal()
        )
        
        # 2. Proyección de output
        self.output_projection = nn.Dense(
            features=self.hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=self.use_bias,
            kernel_init=nn.initializers.lecun_normal()
        )
        
        # 3. Dropout (only if necessary)
        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(
                rate=self.dropout_rate,
                broadcast_dims=(-2,)  # not dropout en heads completos
            )

    @partial(jax.jit, static_argnums=(0, 5))
    def __call__(
        self,
        query: jnp.ndarray,
        key: Optional[jnp.ndarray] = None,
        value: Optional[jnp.ndarray] = None,
        mask: Optional[jnp.ndarray] = None,
        training: bool = False
    ) -> Dict[str, jnp.ndarray]:
        """
        Forward pass optimizado.
        
        Args:
            query: [batch, seq_len, hidden_size]
            key: [batch, seq_len, hidden_size] (optional)
            value: [batch, seq_len, hidden_size] (optional)
            mask: [batch, seq_len, seq_len] (optional)
            training: bool
            
        Returns:
            Dict with output and métricas
        """
        
        # if not se proporcionan key/value, use query (self-attention)
        input_tensor = query if key is None else key
        
        batch_size, seq_len, _ = query.shape
        
        # 1. Proyección QKV fusionada
        if key is None and value is None:
            # Self-attention: use query for all
            qkv = self.qkv_projection(query)
        else:
            # Cross-attention: Q de query, KV de input_tensor
            q = self.qkv_projection(query)[:, :, :self.hidden_size]
            kv = self.qkv_projection(input_tensor)[:, :, self.hidden_size:]
            qkv = jnp.concatenate([q, kv], axis=-1)
        
        # 2. Split and reshape for multi-head
        q, k, v = jnp.split(qkv, 3, axis=-1)
        
        q = self._reshape_for_attention(q, batch_size, seq_len)
        k = self._reshape_for_attention(k, batch_size, seq_len)
        v = self._reshape_for_attention(v, batch_size, seq_len)
        
        # 3. Attention computation optimizada
        attended, attention_weights = self._compute_attention(
            q, k, v, mask, training
        )
        
        # 4. Reshape and proyección end
        attended = self._reshape_from_attention(attended, batch_size, seq_len)
        output = self.output_projection(attended)
        
        # 5. Residual connection (if query es del same size)
        if output.shape == query.shape:
            output = output + query
        
        # 6. Métricas
        metrics = self._compute_metrics(attention_weights, output)
        
        return {
            "output": output,
            "attention_weights": attention_weights,
            "metrics": metrics
        }

    @partial(jax.jit, static_argnums=(0,))
    def _reshape_for_attention(
        self, 
        x: jnp.ndarray, 
        batch_size: int, 
        seq_len: int
    ) -> jnp.ndarray:
        """Reshape for multi-head attention."""
        return x.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

    @partial(jax.jit, static_argnums=(0,))
    def _reshape_from_attention(
        self, 
        x: jnp.ndarray, 
        batch_size: int, 
        seq_len: int
    ) -> jnp.ndarray:
        """Reshape de vuelta since multi-head attention."""
        return x.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.hidden_size)

    @partial(jax.jit, static_argnums=(0, 5))
    def _compute_attention(
        self,
        q: jnp.ndarray,
        k: jnp.ndarray, 
        v: jnp.ndarray,
        mask: Optional[jnp.ndarray],
        training: bool
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute attention de forma eficiente.
        
        Args:
            q, k, v: [batch, num_heads, seq_len, head_dim]
            mask: [batch, seq_len, seq_len] (optional)
            training: bool
            
        Returns:
            (attended_values, attention_weights)
        """
        
        # 1. Compute attention scores
        attention_logits = jnp.einsum('bhqd,bhkd->bhqk', q, k) * self.scale
        
        # 2. Apply mask if se proporciona
        if mask is not None:
            # Expand mask for múltiples heads
            mask = mask[:, None, :, :]  # [batch, 1, seq_len, seq_len]
            # use un value muy negativo instead of -inf for estabilidad
            attention_logits = jnp.where(mask, attention_logits, -1e9)
        
        # 3. Softmax
        attention_weights = jax.nn.softmax(attention_logits, axis=-1)
        
        # 4. Dropout en attention weights (if está entrenando)
        if training and self.dropout_rate > 0:
            attention_weights = self.dropout(attention_weights, deterministic=False)
        
        # 5. Apply attention a values
        attended = jnp.einsum('bhqk,bhkd->bhqd', attention_weights, v)
        
        return attended, attention_weights

    @partial(jax.jit, static_argnums=(0,))
    def _compute_metrics(
        self,
        attention_weights: jnp.ndarray,
        output: jnp.ndarray
    ) -> Dict[str, jnp.ndarray]:
        """Compute métricas de forma eficiente."""
        
        # Flatten attention weights for estadísticas
        attn_flat = attention_weights.reshape(-1, attention_weights.shape[-1])
        
        metrics = {
            # Métricas de attention
            "attention_entropy": jnp.mean(
                -jnp.sum(attn_flat * jnp.log(attn_flat + 1e-8), axis=-1)
            ),
            "attention_max": jnp.max(attention_weights),
            "attention_concentration": jnp.mean(jnp.max(attn_flat, axis=-1)),
            "attention_sparsity": jnp.mean(attn_flat < 0.01),  # % de pesos muy pequeños
            
            # Métricas de output
            "output_norm": jnp.linalg.norm(output),
            "output_mean": jnp.mean(jnp.abs(output)),
            "output_std": jnp.std(output)
        }
        
        return metrics


class MultiScaleSharedAttention(nn.Module):
    """
    Atención multi-escala for capture patrones a diferentes resoluciones.
    Optimizada for tpu v4-32.
    """
    
    hidden_size: int
    num_heads: int
    scales: Tuple[int, ...] = (1, 2, 4, 8)  # Diferentes escalas de atención
    dropout_rate: float = 0.1
    dtype: Any = jnp.bfloat16
    
    def setup(self):
        """Inicializa attention modules for each escala."""
        
        # create attention module for each escala
        self.attention_modules = {}
        for scale in self.scales:
            self.attention_modules[f"scale_{scale}"] = OptimizedSharedAttention(
                hidden_size=self.hidden_size,
                num_heads=max(1, self.num_heads // scale),  # except heads for escalas mayores
                dropout_rate=self.dropout_rate,
                dtype=self.dtype
            )
        
        # Proyección for combine escalas
        self.scale_fusion = nn.Dense(
            features=self.hidden_size,
            dtype=self.dtype,
            use_bias=False
        )
        
        # Gate for ponderar escalas
        self.scale_gate = nn.Dense(
            features=len(self.scales),
            dtype=self.dtype
        )

    @nn.compact
    def __call__(
        self,
        query: jnp.ndarray,
        key: Optional[jnp.ndarray] = None,
        value: Optional[jnp.ndarray] = None,
        mask: Optional[jnp.ndarray] = None,
        training: bool = False
    ) -> Dict[str, jnp.ndarray]:
        """
        Forward pass with atención multi-escala.
        
        Args:
            query: [batch, seq_len, hidden_size]
            key, value: optional for cross-attention
            mask: Máscara de atención
            training: Modo entrenamiento
            
        Returns:
            Dict with output and métricas
        """
        
        batch_size, seq_len, _ = query.shape
        scale_outputs = []
        scale_metrics = {}
        
        # 1. process each escala
        for scale in self.scales:
            # Downsample for escalas > 1
            if scale > 1:
                # Pooling for reduce resolution
                pooled_query = self._adaptive_pool(query, scale)
                pooled_key = self._adaptive_pool(key if key is not None else query, scale)
                pooled_value = self._adaptive_pool(value if value is not None else query, scale)
                pooled_mask = self._downsample_mask(mask, scale) if mask is not None else None
            else:
                pooled_query = query
                pooled_key = key
                pooled_value = value
                pooled_mask = mask
            
            # apply attention
            attn_result = self.attention_modules[f"scale_{scale}"](
                pooled_query, pooled_key, pooled_value, pooled_mask, training
            )
            
            # Upsample de vuelta if es necessary
            if scale > 1:
                output_upsampled = self._adaptive_upsample(
                    attn_result["output"], seq_len
                )
            else:
                output_upsampled = attn_result["output"]
            
            scale_outputs.append(output_upsampled)
            scale_metrics[f"scale_{scale}"] = attn_result["metrics"]
        
        # 2. combine escalas with gating
        stacked_outputs = jnp.stack(scale_outputs, axis=-1)  # [batch, seq_len, hidden, num_scales]
        
        # Compute gates basado en query
        query_pooled = jnp.mean(query, axis=1)  # [batch, hidden]
        scale_weights = nn.softmax(self.scale_gate(query_pooled), axis=-1)  # [batch, num_scales]
        
        # Weighted combination
        combined = jnp.einsum('bshn,bn->bsh', stacked_outputs, scale_weights)
        
        # 3. Proyección end
        output = self.scale_fusion(combined)
        
        # 4. Residual connection
        output = output + query
        
        # 5. Métricas agregadas
        combined_metrics = self._aggregate_metrics(scale_metrics, scale_weights)
        
        return {
            "output": output,
            "scale_weights": scale_weights,
            "metrics": combined_metrics
        }

    @partial(jax.jit, static_argnums=(0, 2))
    def _adaptive_pool(self, x: jnp.ndarray, scale: int) -> jnp.ndarray:
        """Pooling adaptativo for downsampling."""
        if scale == 1:
            return x
        
        batch_size, seq_len, hidden_size = x.shape
        new_seq_len = seq_len // scale
        
        # Reshape and average pool
        x_reshaped = x[:, :new_seq_len * scale, :].reshape(
            batch_size, new_seq_len, scale, hidden_size
        )
        return jnp.mean(x_reshaped, axis=2)

    @partial(jax.jit, static_argnums=(0, 2))
    def _adaptive_upsample(self, x: jnp.ndarray, target_len: int) -> jnp.ndarray:
        """Upsampling adaptativo."""
        batch_size, seq_len, hidden_size = x.shape
        
        if seq_len == target_len:
            return x
        
        # simple repeat upsampling
        scale = target_len // seq_len
        remainder = target_len % seq_len
        
        # Repeat each element
        upsampled = jnp.repeat(x, scale, axis=1)
        
        # Handle remainder by repeating last elements
        if remainder > 0:
            last_elements = x[:, -remainder:, :]
            upsampled = jnp.concatenate([upsampled, last_elements], axis=1)
        
        return upsampled[:, :target_len, :]

    @partial(jax.jit, static_argnums=(0, 2))
    def _downsample_mask(self, mask: jnp.ndarray, scale: int) -> jnp.ndarray:
        """Downsample mask manteniendo structure de atención."""
        if scale == 1:
            return mask
        
        batch_size, seq_len, _ = mask.shape
        new_seq_len = seq_len // scale
        
        # take each scale-ésimo element
        downsampled = mask[:, ::scale, ::scale]
        return downsampled[:, :new_seq_len, :new_seq_len]

    def _aggregate_metrics(
        self, 
        scale_metrics: Dict[str, Dict[str, jnp.ndarray]], 
        scale_weights: jnp.ndarray
    ) -> Dict[str, jnp.ndarray]:
        """Agrega métricas de todas las escalas."""
        
        aggregated = {}
        
        # obtain todas las métricas únicas
        all_metric_keys = set()
        for metrics in scale_metrics.values():
            all_metric_keys.update(metrics.keys())
        
        # add each métrica
        for metric_key in all_metric_keys:
            metric_values = []
            weights = []
            
            for i, scale in enumerate(self.scales):
                scale_key = f"scale_{scale}"
                if metric_key in scale_metrics[scale_key]:
                    metric_values.append(scale_metrics[scale_key][metric_key])
                    weights.append(scale_weights[:, i])
            
            if metric_values:
                # Weighted average across scales
                stacked_values = jnp.stack(metric_values, axis=0)  # [num_scales, ...]
                stacked_weights = jnp.stack(weights, axis=0)  # [num_scales, batch]
                
                # Global average across batch and scales
                aggregated[f"avg_{metric_key}"] = jnp.mean(stacked_values)
                aggregated[f"weighted_{metric_key}"] = jnp.sum(
                    stacked_values * jnp.mean(stacked_weights, axis=1, keepdims=True)
                )
        
        # Métricas específicas de multi-escala
        aggregated["scale_entropy"] = -jnp.mean(
            jnp.sum(scale_weights * jnp.log(scale_weights + 1e-8), axis=-1)
        )
        aggregated["scale_concentration"] = jnp.mean(jnp.max(scale_weights, axis=-1))
        
        return aggregated


class EfficiencyOptimizedAttention(nn.Module):
    """
    Atención optimizada for máxima eficiencia en tpu.
    Reduce complejidad de or(n²) a or(n log n) for secuencias largas.
    """
    
    hidden_size: int
    num_heads: int
    window_size: int = 512  # Ventana deslizante
    num_random_chunks: int = 64  # Chunks aleatorios for conectividad global
    dropout_rate: float = 0.1
    dtype: Any = jnp.bfloat16
    
    def setup(self):
        """initialization for attention eficiente."""
        
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Proyecciones QKV
        self.qkv_projection = nn.Dense(
            features=3 * self.hidden_size,
            dtype=self.dtype,
            use_bias=False,
            kernel_init=nn.initializers.lecun_normal()
        )
        
        # Proyección de output
        self.output_projection = nn.Dense(
            features=self.hidden_size,
            dtype=self.dtype,
            use_bias=False
        )
        
        # Dropout
        self.dropout = nn.Dropout(rate=self.dropout_rate)

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        training: bool = False
    ) -> Dict[str, jnp.ndarray]:
        """
        Attention eficiente with complejidad reducida.
        
        Args:
            x: [batch, seq_len, hidden_size]
            mask: optional
            training: bool
            
        Returns:
            Dict with output and métricas
        """
        
        batch_size, seq_len, _ = x.shape
        
        # 1. Proyección QKV
        qkv = self.qkv_projection(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        
        # 2. Reshape for multi-head
        q = self._reshape_for_heads(q)
        k = self._reshape_for_heads(k)
        v = self._reshape_for_heads(v)
        
        # 3. apply attention eficiente
        if seq_len <= self.window_size:
            # for secuencias cortas, use attention completa
            attended = self._full_attention(q, k, v, mask, training)
        else:
            # for secuencias largas, use attention eficiente
            attended = self._efficient_attention(q, k, v, mask, training)
        
        # 4. Reshape and proyección end
        attended = self._reshape_from_heads(attended)
        output = self.output_projection(attended)
        
        # 5. Residual connection
        output = output + x
        
        # 6. Métricas
        metrics = {
            "sequence_length": seq_len,
            "attention_type": "full" if seq_len <= self.window_size else "efficient",
            "output_norm": jnp.linalg.norm(output),
            "compression_ratio": min(1.0, self.window_size / seq_len)
        }
        
        return {
            "output": output,
            "metrics": metrics
        }

    def _reshape_for_heads(self, x: jnp.ndarray) -> jnp.ndarray:
        """Reshape for multi-head attention."""
        batch_size, seq_len, _ = x.shape
        return x.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

    def _reshape_from_heads(self, x: jnp.ndarray) -> jnp.ndarray:
        """Reshape de vuelta since multi-head."""
        batch_size, _, seq_len, _ = x.shape
        return x.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.hidden_size)

    def _full_attention(
        self,
        q: jnp.ndarray,
        k: jnp.ndarray,
        v: jnp.ndarray,
        mask: Optional[jnp.ndarray],
        training: bool
    ) -> jnp.ndarray:
        """Attention completa for secuencias cortas."""
        
        # Standard scaled dot-product attention
        attention_logits = jnp.einsum('bhqd,bhkd->bhqk', q, k) * self.scale
        
        if mask is not None:
            attention_logits = jnp.where(mask[:, None, :, :], attention_logits, -1e9)
        
        attention_weights = jax.nn.softmax(attention_logits, axis=-1)
        
        if training:
            attention_weights = self.dropout(attention_weights, deterministic=False)
        
        return jnp.einsum('bhqk,bhkd->bhqd', attention_weights, v)

    def _efficient_attention(
        self,
        q: jnp.ndarray,
        k: jnp.ndarray,
        v: jnp.ndarray,
        mask: Optional[jnp.ndarray],
        training: bool
    ) -> jnp.ndarray:
        """
        Attention eficiente for secuencias largas.
        Combina attention local (ventana deslizante) with conexiones globales aleatorias.
        """
        
        batch_size, num_heads, seq_len, head_dim = q.shape
        
        # 1. Local attention (ventana deslizante)
        local_attended = self._sliding_window_attention(q, k, v, training)
        
        # 2. Global attention (chunks aleatorios)
        global_attended = self._random_chunk_attention(q, k, v, training)
        
        # 3. combine local and global
        # use gate for ponderar between local and global
        gate_logits = jnp.mean(q, axis=(2, 3))  # [batch, num_heads]
        gate_weights = jax.nn.sigmoid(gate_logits)[:, :, None, None]
        
        attended = gate_weights * global_attended + (1 - gate_weights) * local_attended
        
        return attended

    def _sliding_window_attention(
        self,
        q: jnp.ndarray,
        k: jnp.ndarray,
        v: jnp.ndarray,
        training: bool
    ) -> jnp.ndarray:
        """Attention with ventana deslizante."""
        
        batch_size, num_heads, seq_len, head_dim = q.shape
        window = self.window_size
        
        # Padding for ventanas completas
        pad_len = window - (seq_len % window) if seq_len % window != 0 else 0
        
        if pad_len > 0:
            q_padded = jnp.pad(q, ((0, 0), (0, 0), (0, pad_len), (0, 0)))
            k_padded = jnp.pad(k, ((0, 0), (0, 0), (0, pad_len), (0, 0)))
            v_padded = jnp.pad(v, ((0, 0), (0, 0), (0, pad_len), (0, 0)))
        else:
            q_padded = q
            k_padded = k
            v_padded = v
        
        padded_len = q_padded.shape[2]
        num_windows = padded_len // window
        
        # Reshape for process ventanas en paralelo
        q_windows = q_padded.reshape(batch_size, num_heads, num_windows, window, head_dim)
        k_windows = k_padded.reshape(batch_size, num_heads, num_windows, window, head_dim)
        v_windows = v_padded.reshape(batch_size, num_heads, num_windows, window, head_dim)
        
        # Attention inside de each ventana
        attention_logits = jnp.einsum('bhnqd,bhnkd->bhnqk', q_windows, k_windows) * self.scale
        attention_weights = jax.nn.softmax(attention_logits, axis=-1)
        
        if training:
            attention_weights = self.dropout(attention_weights, deterministic=False)
        
        attended_windows = jnp.einsum('bhnqk,bhnkd->bhnqd', attention_weights, v_windows)
        
        # Reshape de vuelta
        attended = attended_windows.reshape(batch_size, num_heads, padded_len, head_dim)
        
        # remove padding
        if pad_len > 0:
            attended = attended[:, :, :seq_len, :]
        
        return attended

    def _random_chunk_attention(
        self,
        q: jnp.ndarray,
        k: jnp.ndarray,
        v: jnp.ndarray,
        training: bool
    ) -> jnp.ndarray:
        """Attention with chunks aleatorios for conectividad global."""
        
        batch_size, num_heads, seq_len, head_dim = q.shape
        chunk_size = seq_len // self.num_random_chunks
        
        if chunk_size < 1:
            # if hay more chunks que tokens, use attention completa
            return self._full_attention(q, k, v, None, training)
        
        # select chunks aleatorios
        rng = self.make_rng('random') if training else jax.random.PRNGKey(42)
        chunk_indices = jax.random.choice(
            rng, 
            seq_len, 
            shape=(self.num_random_chunks,), 
            replace=False
        )
        chunk_indices = jnp.sort(chunk_indices)
        
        # Extraer chunks
        q_chunks = q[:, :, chunk_indices, :]
        k_chunks = k[:, :, chunk_indices, :]
        v_chunks = v[:, :, chunk_indices, :]
        
        # Attention between chunks
        attention_logits = jnp.einsum('bhqd,bhkd->bhqk', q_chunks, k_chunks) * self.scale
        attention_weights = jax.nn.softmax(attention_logits, axis=-1)
        
        if training:
            attention_weights = self.dropout(attention_weights, deterministic=False)
        
        attended_chunks = jnp.einsum('bhqk,bhkd->bhqd', attention_weights, v_chunks)
        
        # Interpolar de vuelta a secuencia completa
        attended = jnp.zeros_like(q)
        attended = attended.at[:, :, chunk_indices, :].set(attended_chunks)
        
        # Interpolación simple for posiciones not cubiertas
        for i in range(seq_len):
            if i not in chunk_indices:
                # find chunks more cercanos
                distances = jnp.abs(chunk_indices - i)
                closest_idx = jnp.argmin(distances)
                attended = attended.at[:, :, i, :].set(attended_chunks[:, :, closest_idx, :])
        
        return attended


# Funciones de utilidad
def create_shared_attention(
    hidden_size: int,
    num_heads: int,
    attention_type: str = "standard",
    **kwargs
) -> nn.Module:
    """
    Factory function para crear diferentes tipos de attention.
    
    Args:
        hidden_size: Dimensión del espacio oculto
        num_heads: Número de cabezas de atención
        attention_type: Tipo de attention ("standard", "multiscale", "efficient")
        **kwargs: Argumentos adicionales
    
    Returns:
        Módulo de attention configurado
    """
    
    if attention_type == "standard":
        return OptimizedSharedAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            **kwargs
        )
    elif attention_type == "multiscale":
        return MultiScaleSharedAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            **kwargs
        )
    elif attention_type == "efficient":
        return EfficiencyOptimizedAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown attention type: {attention_type}")


# Benchmark functions
@jax.jit
def benchmark_attention_performance(
    attention_module: nn.Module,
    params: Dict,
    batch_size: int = 8,
    seq_len: int = 2048,
    hidden_size: int = 768,
    num_iterations: int = 50
) -> Dict[str, float]:
    """Benchmark de rendimiento de attention."""
    
    # data de test
    rng = jax.random.PRNGKey(42)
    x = jax.random.normal(rng, (batch_size, seq_len, hidden_size))
    
    # function a medir
    def forward_fn():
        return attention_module.apply(params, x, training=True)
    
    # Warmup
    for _ in range(5):
        forward_fn()
    
    # Benchmark
    import time
    start_time = time.time()
    
    for _ in range(num_iterations):
        result = forward_fn()
        result["output"].block_until_ready()
    
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time = total_time / num_iterations
    tokens_per_sec = (batch_size * seq_len) / avg_time
    
    return {
        "avg_time_ms": avg_time * 1000,
        "tokens_per_second": tokens_per_sec,
        "memory_efficiency": seq_len / (seq_len ** 2),  # Theoretical efficiency
        "total_time_sec": total_time
    }


if __name__ == "__main__":
    # Test de los diferentes tipos de attention
    hidden_size = 768
    num_heads = 12
    
    # Standard attention
    std_attention = create_shared_attention(hidden_size, num_heads, "standard")
    
    # Multi-scale attention
    ms_attention = create_shared_attention(hidden_size, num_heads, "multiscale")
    
    # Efficient attention
    eff_attention = create_shared_attention(hidden_size, num_heads, "efficient")
    
    print("✅ Todos los módulos de attention creados exitosamente")
    
    # Test basic
    rng = jax.random.PRNGKey(42)
    x = jax.random.normal(rng, (2, 128, hidden_size))
    
    # Inicializar parámetros
    std_params = std_attention.init(rng, x, training=False)
    
    # Forward pass
    result = std_attention.apply(std_params, x, training=False)
    
    print(f"Output shape: {result['output'].shape}")
    print(f"Metrics: {list(result['metrics'].keys())}")
