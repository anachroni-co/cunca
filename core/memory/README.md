# Core Memory Module

**Multi-Scale Continuum Memory System for Long-Context Understanding**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CONTINUUM MEMORY ARCHITECTURE                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Input Stream                                                                │
│      │                                                                       │
│      ▼                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                     Memory Bank Hierarchy                                ││
│  │                                                                          ││
│  │  ┌─────────────────┐  Update: Every Step                                ││
│  │  │  Ultra-Short    │  Capacity: 16 tokens                               ││
│  │  │  Memory Bank    │  Purpose: Immediate context                        ││
│  │  └────────┬────────┘                                                    ││
│  │           │ Consolidation                                                ││
│  │           ▼                                                              ││
│  │  ┌─────────────────┐  Update: Every 10 Steps                            ││
│  │  │    Short-Term   │  Capacity: 256 tokens                              ││
│  │  │  Memory Bank    │  Purpose: Recent context                           ││
│  │  └────────┬────────┘                                                    ││
│  │           │ Consolidation                                                ││
│  │           ▼                                                              ││
│  │  ┌─────────────────┐  Update: Every 100 Steps                           ││
│  │  │   Medium-Term   │  Capacity: 2048 tokens                             ││
│  │  │  Memory Bank    │  Purpose: Session context                          ││
│  │  └────────┬────────┘                                                    ││
│  │           │ Consolidation                                                ││
│  │           ▼                                                              ││
│  │  ┌─────────────────┐  Update: Every 1000 Steps                          ││
│  │  │    Long-Term    │  Capacity: 8192 tokens                             ││
│  │  │  Memory Bank    │  Purpose: Knowledge base                           ││
│  │  └─────────────────┘                                                    ││
│  │                                                                          ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  Cross-Temporal Attention                                                    │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  Query (current) attends to all memory banks simultaneously             ││
│  │  ───────────────────────────────────────────────────────────            ││
│  │       ↓           ↓              ↓               ↓                      ││
│  │  [Ultra-Short] [Short]      [Medium]        [Long-Term]                 ││
│  │       ↓           ↓              ↓               ↓                      ││
│  │  ──────────────── Weighted Fusion ────────────────────                  ││
│  │                         │                                                ││
│  │                         ▼                                                ││
│  │                    [Output]                                              ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Overview

The memory module implements a Continuum Memory System inspired by the HOPE architecture from Nested Learning (Behrouz et al., NeurIPS 2025). The key insight is managing memory at multiple time scales with different update frequencies to avoid catastrophic forgetting and enable long-context understanding.

## Key Concepts

### Nested Learning Principle

Different memory components update at different frequencies, separating concerns across time scales:

| Memory Bank | Capacity | Update Frequency | Purpose |
|-------------|----------|------------------|---------|
| Ultra-Short | 16 tokens | Every step | Immediate attention context |
| Short-Term | 256 tokens | Every 10 steps | Recent conversational context |
| Medium-Term | 2048 tokens | Every 100 steps | Session-level understanding |
| Long-Term | 8192 tokens | Every 1000 steps | Persistent knowledge base |

### Memory Consolidation

Information flows from fast-updating to slow-updating banks through consolidation:

```
Ultra-Short → Short-Term → Medium-Term → Long-Term
     │             │             │            │
   (fast)      (medium)       (slow)     (very slow)
```

This mimics biological memory consolidation where important information is gradually transferred to long-term storage.

## Module Structure

```
core/memory/
├── __init__.py              # Module exports
└── continuum_memory.py      # Main implementation
```

## Quick Start

### Basic Usage

```python
from core.memory import ContinuumMemorySystem, ContinuumMemoryConfig

# Create memory system
config = ContinuumMemoryConfig(
    hidden_size=768,
    num_heads=12,
    ultra_short_size=16,
    short_term_size=256,
    medium_term_size=2048,
    long_term_size=8192
)

memory = ContinuumMemorySystem(config)

# Process input with memory
output, memory_state = memory.forward(
    input_embeddings=embeddings,
    previous_state=None  # or memory_state from previous call
)

# Memory state persists across calls
output2, memory_state = memory.forward(
    input_embeddings=new_embeddings,
    previous_state=memory_state
)
```

### Integration with Model

```python
class CapibaraWithMemory(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformer = TransformerModel(config)
        self.memory = ContinuumMemorySystem(
            ContinuumMemoryConfig.from_model_config(config)
        )
        self.memory_state = None

    def forward(self, input_ids, attention_mask=None):
        # Get embeddings
        embeddings = self.transformer.embed(input_ids)

        # Enhance with memory
        memory_output, self.memory_state = self.memory(
            input_embeddings=embeddings,
            previous_state=self.memory_state
        )

        # Continue with transformer
        output = self.transformer.forward_with_memory(
            embeddings=embeddings,
            memory_context=memory_output
        )

        return output

    def reset_memory(self):
        """Reset memory state for new conversation."""
        self.memory_state = None
```

## Components

### ContinuumMemoryConfig

Configuration for the memory system.

```python
from core.memory import ContinuumMemoryConfig

config = ContinuumMemoryConfig(
    # Model dimensions
    hidden_size=768,
    num_heads=12,
    head_dim=64,

    # Memory bank sizes
    ultra_short_size=16,
    short_term_size=256,
    medium_term_size=2048,
    long_term_size=8192,

    # Update frequencies
    ultra_short_update_freq=1,    # Every step
    short_term_update_freq=10,    # Every 10 steps
    medium_term_update_freq=100,  # Every 100 steps
    long_term_update_freq=1000,   # Every 1000 steps

    # Consolidation settings
    consolidation_ratio=0.1,      # 10% of entries consolidate
    decay_rate=0.99,              # Memory decay factor

    # Attention settings
    cross_temporal_attention=True,
    memory_dropout=0.1
)
```

### MemoryBank

Individual memory bank component.

```python
from core.memory import MemoryBank, MemoryBankConfig

# Create memory bank
bank_config = MemoryBankConfig(
    name="short_term",
    capacity=256,
    hidden_size=768,
    update_frequency=10
)

bank = MemoryBank(bank_config)

# Store entry
bank.store(
    key=key_embedding,
    value=value_embedding,
    importance=0.8
)

# Retrieve relevant memories
retrieved = bank.retrieve(
    query=query_embedding,
    top_k=10
)

# Decay old memories
bank.apply_decay(decay_rate=0.99)
```

### CrossTemporalAttention

Attention mechanism across all memory banks.

```python
from core.memory import CrossTemporalAttention

attention = CrossTemporalAttention(
    hidden_size=768,
    num_heads=12,
    num_banks=4
)

# Query attends to all memory banks
output = attention(
    query=current_embeddings,
    memory_banks=[ultra_short, short_term, medium_term, long_term]
)
```

## Memory Operations

### Storing Memories

```python
# Automatic storage during forward pass
output, state = memory.forward(embeddings, state)

# Manual storage with importance scoring
memory.store_explicit(
    embeddings=important_embeddings,
    importance_scores=scores,  # [0, 1] importance
    target_bank="medium_term"
)
```

### Retrieving Memories

```python
# Query-based retrieval
retrieved = memory.retrieve(
    query=query_embedding,
    top_k=20,
    banks=["short_term", "medium_term", "long_term"]
)

# Get full memory contents
all_memories = memory.get_all_memories()
# {
#     "ultra_short": [...],
#     "short_term": [...],
#     "medium_term": [...],
#     "long_term": [...]
# }
```

### Memory Consolidation

```python
# Manual consolidation trigger
memory.consolidate()

# Consolidation happens automatically based on update frequencies
# During forward pass, step counter triggers consolidation:
#   Step 10: short_term consolidates from ultra_short
#   Step 100: medium_term consolidates from short_term
#   Step 1000: long_term consolidates from medium_term
```

### Memory Decay

```python
# Apply decay to reduce memory salience over time
memory.apply_decay()

# Custom decay rates per bank
memory.apply_decay(
    decay_rates={
        "ultra_short": 0.9,   # Fast decay
        "short_term": 0.95,
        "medium_term": 0.99,
        "long_term": 0.999   # Very slow decay
    }
)
```

## Advanced Usage

### Persistent Memory Across Sessions

```python
# Save memory state
memory_state = memory.get_state()
save_to_disk(memory_state, "memory_checkpoint.pt")

# Load in new session
memory_state = load_from_disk("memory_checkpoint.pt")
memory.load_state(memory_state)
```

### Memory Analysis

```python
# Get memory statistics
stats = memory.get_statistics()
# {
#     "ultra_short": {"size": 16, "utilization": 1.0, "avg_importance": 0.7},
#     "short_term": {"size": 200, "utilization": 0.78, "avg_importance": 0.6},
#     "medium_term": {"size": 1500, "utilization": 0.73, "avg_importance": 0.5},
#     "long_term": {"size": 5000, "utilization": 0.61, "avg_importance": 0.4}
# }

# Visualize memory attention patterns
attention_weights = memory.get_attention_weights()
visualize_memory_attention(attention_weights)
```

### Custom Memory Banks

```python
from core.memory import ContinuumMemorySystem, MemoryBankConfig

# Add custom memory bank for specialized content
memory.add_bank(
    MemoryBankConfig(
        name="code_memory",
        capacity=1024,
        hidden_size=768,
        update_frequency=50,
        retrieval_strategy="semantic"
    )
)
```

## Performance Considerations

### Memory Efficiency

```
┌────────────────────────────────────────────────────────────────┐
│                  Memory Footprint                              │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Bank          │ Tokens │ Hidden │ Total Memory              │
│  ─────────────────────────────────────────────────────────── │
│  Ultra-Short   │    16  │   768  │   ~50 KB                  │
│  Short-Term    │   256  │   768  │  ~800 KB                  │
│  Medium-Term   │  2048  │   768  │  ~6.3 MB                  │
│  Long-Term     │  8192  │   768  │ ~25.2 MB                  │
│  ─────────────────────────────────────────────────────────── │
│  Total         │ 10512  │   768  │ ~32.4 MB                  │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Computational Cost

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Store | O(1) | Ring buffer insertion |
| Retrieve | O(n × d) | n = bank size, d = hidden |
| Cross-Attention | O(m × n × d) | m = query, n = total memory |
| Consolidation | O(k × d) | k = consolidation batch |

## References

This implementation is inspired by:

- **Nested Learning / HOPE Architecture**: Behrouz, A.; Razaviyayn, M.; Mirrokni, V.; Zhong, P. (2025). "Nested Learning: The Illusion of Deep Learning Architectures" - NeurIPS 2025
- **Memory Consolidation Theory**: Complementary Learning Systems (CLS) theory in cognitive neuroscience
- **Multi-Scale Memory**: Transformer-XL and related work on recurrence in transformers

## See Also

- [Core Module](../README.md)
- [Attention Mechanisms](../attention/README.md)
- [Chain-of-Thought](../cot/README.md)
