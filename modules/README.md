# Modules

**Specialized Processing Modules and Adaptive Routing System**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        MODULE ARCHITECTURE                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                          Input Sequence                                      │
│                               │                                              │
│                               ▼                                              │
│                    ┌─────────────────────┐                                  │
│                    │   Adaptive Router   │                                  │
│                    │   ┌─────────────┐   │                                  │
│                    │   │  Analysis   │   │                                  │
│                    │   │  Scoring    │   │                                  │
│                    │   └─────────────┘   │                                  │
│                    └──────────┬──────────┘                                  │
│                               │                                              │
│         ┌─────────────────────┼─────────────────────┐                       │
│         │                     │                     │                       │
│         ▼                     ▼                     ▼                       │
│  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐                 │
│  │  Reasoning  │      │  Attention  │      │ Specialized │                 │
│  │   Module    │      │   Module    │      │ Processors  │                 │
│  └─────────────┘      └─────────────┘      └─────────────┘                 │
│         │                     │                     │                       │
│         └─────────────────────┴─────────────────────┘                       │
│                               │                                              │
│                               ▼                                              │
│                    ┌─────────────────────┐                                  │
│                    │   Output Fusion     │                                  │
│                    └─────────────────────┘                                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Overview

The modules package contains specialized processing components that handle different aspects of the model's computation. This includes adaptive routing, shared attention mechanisms, hierarchical reasoning, and domain-specific processors.

## Module Structure

```
modules/
├── __init__.py                     # Module exports
├── capibara_adaptive_router.py     # Adaptive routing system
├── shared_attention.py             # Shared attention mechanisms
├── hierarchical_reasoning.py       # Hierarchical reasoning module
├── specialized_processors.py       # Domain-specific processors
├── ultra_module_orchestrator.py    # Module orchestration system
├── ultra_modules_demo.py           # Demo and examples
├── analysis_modules_and_jax_decorators.md  # Documentation
└── personality/                    # Personality modules
    └── ...
```

## Components

### Adaptive Router

Dynamically routes inputs to appropriate processing modules based on content analysis.

```python
from modules import AdaptiveRouter

router = AdaptiveRouter(
    modules=["reasoning", "attention", "specialized"],
    routing_strategy="learned",  # or "rule_based", "hybrid"
    top_k=2  # Activate top 2 modules
)

# Route input
routing_weights = router.route(input_embeddings)
# {"reasoning": 0.6, "attention": 0.3, "specialized": 0.1}

# Apply weighted processing
output = router.forward(input_embeddings)
```

**Routing Strategies:**

```
┌────────────────────────────────────────────────────────────────┐
│                    Routing Strategies                          │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Rule-Based:                                                   │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ IF contains_math(input) → route to reasoning_module     │ │
│  │ IF is_long_context(input) → route to attention_module   │ │
│  │ IF is_code(input) → route to code_processor             │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                │
│  Learned (Neural):                                            │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ router_logits = Linear(input_embeddings)                │ │
│  │ routing_weights = softmax(router_logits / temperature)  │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                │
│  Hybrid:                                                       │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ rule_weights = apply_rules(input)                       │ │
│  │ learned_weights = neural_router(input)                  │ │
│  │ final_weights = combine(rule_weights, learned_weights)  │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Shared Attention

Efficient attention mechanisms shared across model layers.

```python
from modules import SharedAttention, AttentionConfig

# Configure shared attention
config = AttentionConfig(
    num_heads=16,
    head_dim=64,
    num_shared_layers=4,
    share_weights=True,
    use_flash_attention=True
)

attention = SharedAttention(config)

# Forward pass
output = attention(
    query=query_states,
    key=key_states,
    value=value_states,
    attention_mask=mask
)
```

**Benefits of Shared Attention:**

| Aspect | Standard | Shared |
|--------|----------|--------|
| Parameters | O(L × d²) | O(d²) |
| Memory | High | Reduced |
| Training | Slower | Faster |
| Generalization | Per-layer | Cross-layer |

### Hierarchical Reasoning

Multi-level reasoning module for complex problem-solving.

```python
from modules import HierarchicalReasoning

reasoning = HierarchicalReasoning(
    num_levels=3,
    hidden_size=768,
    reasoning_steps=5
)

# Perform hierarchical reasoning
result = reasoning.forward(
    input_embeddings=embeddings,
    problem_context=context,
    max_depth=3
)

# Access reasoning trace
trace = result.reasoning_trace
# [
#     {"level": 0, "step": 1, "thought": "..."},
#     {"level": 0, "step": 2, "thought": "..."},
#     {"level": 1, "step": 1, "thought": "..."},
#     ...
# ]
```

**Reasoning Hierarchy:**

```
┌─────────────────────────────────────────────────────────────┐
│                  Hierarchical Reasoning                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Level 0: Surface Understanding                              │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ • Parse input structure                                 ││
│  │ • Identify key entities                                 ││
│  │ • Extract explicit information                          ││
│  └─────────────────────────────────────────────────────────┘│
│                          │                                   │
│                          ▼                                   │
│  Level 1: Relational Reasoning                              │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ • Build relationship graphs                             ││
│  │ • Identify causal chains                                ││
│  │ • Connect related concepts                              ││
│  └─────────────────────────────────────────────────────────┘│
│                          │                                   │
│                          ▼                                   │
│  Level 2: Abstract Reasoning                                │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ • Apply logical rules                                   ││
│  │ • Generate hypotheses                                   ││
│  │ • Synthesize conclusions                                ││
│  └─────────────────────────────────────────────────────────┘│
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Specialized Processors

Domain-specific processing modules.

```python
from modules.specialized_processors import (
    MathProcessor,
    CodeProcessor,
    LanguageProcessor,
    ScienceProcessor
)

# Math processing
math_proc = MathProcessor()
math_result = math_proc.process(
    input_text="Solve: 2x + 5 = 13",
    show_steps=True
)

# Code processing
code_proc = CodeProcessor(
    languages=["python", "javascript", "rust"]
)
code_result = code_proc.process(
    code="def fibonacci(n): ...",
    task="explain"  # or "optimize", "debug", "translate"
)
```

**Available Processors:**

| Processor | Domain | Capabilities |
|-----------|--------|--------------|
| MathProcessor | Mathematics | Symbolic computation, proofs |
| CodeProcessor | Programming | Analysis, generation, debugging |
| LanguageProcessor | NLP | Translation, summarization |
| ScienceProcessor | Science | Scientific reasoning, citations |
| CreativeProcessor | Creative | Story generation, style transfer |

### Module Orchestrator

Coordinates multiple modules for complex tasks.

```python
from modules import ModuleOrchestrator

orchestrator = ModuleOrchestrator(
    modules={
        "router": AdaptiveRouter(...),
        "attention": SharedAttention(...),
        "reasoning": HierarchicalReasoning(...),
        "processors": [MathProcessor(), CodeProcessor()]
    },
    execution_strategy="parallel"  # or "sequential", "adaptive"
)

# Process complex input
result = orchestrator.process(
    input_data=input_embeddings,
    task_type="problem_solving",
    return_intermediates=True
)
```

## Personality Module

Customizable response personality system.

```python
from modules.personality import PersonalityModule, PersonalityConfig

config = PersonalityConfig(
    tone="friendly",           # friendly, professional, casual
    verbosity="balanced",      # concise, balanced, detailed
    formality="medium",        # low, medium, high
    creativity=0.7,            # 0.0 - 1.0
    emoji_usage=False
)

personality = PersonalityModule(config)

# Apply personality to response
styled_response = personality.apply(
    base_response="Here is the answer.",
    context=conversation_context
)
```

## Usage Example

```python
from modules import (
    AdaptiveRouter,
    SharedAttention,
    HierarchicalReasoning,
    ModuleOrchestrator
)

# Build module pipeline
router = AdaptiveRouter(num_modules=4)
attention = SharedAttention(num_heads=16)
reasoning = HierarchicalReasoning(num_levels=3)

orchestrator = ModuleOrchestrator(
    modules={"router": router, "attention": attention, "reasoning": reasoning}
)

# Process input
class ModularModel:
    def __init__(self):
        self.orchestrator = orchestrator

    def forward(self, input_ids, attention_mask):
        # Get embeddings
        embeddings = self.embed(input_ids)

        # Route and process
        output = self.orchestrator.process(
            embeddings,
            attention_mask=attention_mask
        )

        return output
```

## Configuration

```yaml
# config/modules.yaml
modules:
  router:
    strategy: "learned"
    top_k: 2
    temperature: 1.0

  attention:
    num_heads: 16
    head_dim: 64
    use_flash: true
    dropout: 0.1

  reasoning:
    num_levels: 3
    max_steps: 10
    early_stopping: true

  orchestrator:
    execution: "adaptive"
    timeout_ms: 1000
    fallback_module: "attention"
```

## See Also

- [Core Module](../core/README.md)
- [MoE System](../core/moe/README.md)
- [Chain-of-Thought](../core/cot/README.md)
