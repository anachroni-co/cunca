# Production Profiles (TPU)

Merged profiles have been created:
- `fusion_base.toml`: common base with model parameters, memory, router, modules, services, etc.
- `fusion_production.toml`: production profile (TPU v6e / MoE) that complements the base.

Recommended: in code, point to `fusion_production.toml` or `fusion_base.toml` as needed.

Previous files can be kept for compatibility, but migration is suggested.
