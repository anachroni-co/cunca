# Production Profiles (TPU)

Merged profiles have been created:
- `fusion_base.toml`: common base with model parameters, memory, router, modules, services, etc.
- `fusion_production.toml`: production profile (TPU v6e / MoE) that complements the base.

Recommended: in code, point to `fusion_production.toml` or `fusion_base.toml` as needed.

Previous files can be kept for compatibility, but migration is suggested.

## Example

```python
from pathlib import Path
import toml

base_path = Path(__file__).parent
base_config = toml.loads((base_path / "fusion_base.toml").read_text())
prod_config = toml.loads((base_path / "fusion_production.toml").read_text())

merged_config = {**base_config, **prod_config}
print(f"Loaded production profile with {len(merged_config)} top-level keys")
```
