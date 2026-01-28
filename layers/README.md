# Layers Module

This folder contains neural network layer utilities used by CapibaraGPT.

## Current Structure

```
layers/
├── abstract_reasoning/  # Platonic, Quineana, GameTheory
├── pasive/              # Synthetic embedding + distributed attention
├── sparsity/            # BitNet, sparse layers, quantization
├── __init__.py          # Layer exports and availability flags
```

## Notes

- The following uploaded files were removed because they are not part of the
  current layer inventory: `base.py`, `conv1d_block.py`, `embedding.py`,
  `meta_la.py`, `neuro_adaptive.py`, `neurogenesis.py`, `self_attention.py`,
  `smb_layer.py`, `ssm_hybrid_layers.py`, `stack.py`, `ultra_layer_integration.py`.
- If you need any of those layers restored, please provide the source or
  confirm which module should be reinstated.
