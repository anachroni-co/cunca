"""Tests to enforce code factorization standards.

These tests scan source files for patterns that should use the
centralized utilities instead of inline boilerplate.  They act as
guardrails so that new code follows the established conventions in:

- ``layers/jax_compat.py``   – JAX/Flax import guard
- ``core/import_utils.py``   – ``safe_import`` helper
- ``core/backends/lazy_import.py`` – ``ensure_torch`` / ``ensure_jax``
- ``layers/attention_utils.py``  – ``split_heads`` / ``merge_heads``
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import List, Tuple

import pytest

ROOT = Path(__file__).resolve().parents[2]

# ── directories that are checked ────────────────────────────────
LAYER_DIR = ROOT / "layers"
CORE_DIR = ROOT / "core"
BACKEND_DIR = CORE_DIR / "backends"

# ── files that are allowed to keep legacy patterns ──────────────
ALLOWED_JAX_IMPORT_FILES = {
    str(LAYER_DIR / "jax_compat.py"),
    str(CORE_DIR / "backends" / "lazy_import.py"),
    str(CORE_DIR / "backends" / "tpu_backend.py"),
}

ALLOWED_TORCH_IMPORT_FILES = {
    str(CORE_DIR / "backends" / "lazy_import.py"),
    str(CORE_DIR / "backends" / "gpu_backend.py"),
    str(CORE_DIR / "backends" / "utils.py"),
}

ALLOWED_SAFE_IMPORT_FILES = {
    str(CORE_DIR / "import_utils.py"),
}

ALLOWED_ATTENTION_RESHAPE_FILES = {
    str(LAYER_DIR / "attention_utils.py"),
    str(LAYER_DIR / "sparsity" / "sparse_capibara.py"),
}


def _py_files(directory: Path) -> List[Path]:
    """Yield all .py files under *directory*, excluding __pycache__."""
    return sorted(
        p for p in directory.rglob("*.py")
        if "__pycache__" not in str(p)
    )


# ─────────────────────────────────────────────────────────────────
# 1. No duplicate JAX/Flax import guards in layers/
# ─────────────────────────────────────────────────────────────────
_JAX_IMPORT_RE = re.compile(
    r"try\s*:\s*\n\s+import\s+jax\b"
    r"|try\s*:\s*\n\s+from\s+jax\b"
    r"|try\s*:\s*\n\s+import\s+flax\b"
    r"|try\s*:\s*\n\s+from\s+flax\b",
    re.MULTILINE,
)


class TestNoInlineJAXImportsInLayers:
    """New layer files must use ``layers.jax_compat`` instead of
    inline try/except JAX imports."""

    def _violating_files(self) -> List[Tuple[str, int]]:
        violations = []
        for path in _py_files(LAYER_DIR):
            if str(path) in ALLOWED_JAX_IMPORT_FILES:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            matches = list(_JAX_IMPORT_RE.finditer(text))
            if matches:
                line = text[: matches[0].start()].count("\n") + 1
                violations.append((str(path.relative_to(ROOT)), line))
        return violations

    def test_no_inline_jax_imports(self):
        """layers/ files should import from jax_compat, not inline try/except."""
        violations = self._violating_files()
        if violations:
            msg = "Found inline JAX try/except imports (use layers.jax_compat):\n"
            msg += "\n".join(f"  {f}:{line}" for f, line in violations)
            pytest.fail(msg)


# ─────────────────────────────────────────────────────────────────
# 2. No duplicate torch lazy-import pattern in backends/
# ─────────────────────────────────────────────────────────────────
_TORCH_LAZY_RE = re.compile(
    r"try\s*:\s*\n\s+import\s+torch\b",
    re.MULTILINE,
)


class TestNoInlineTorchImportsInBackends:
    """Backend files must use ``core.backends.lazy_import.ensure_torch``
    instead of writing their own try/except for torch."""

    def _violating_files(self) -> List[Tuple[str, int]]:
        violations = []
        for path in _py_files(BACKEND_DIR):
            if str(path) in ALLOWED_TORCH_IMPORT_FILES:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            matches = list(_TORCH_LAZY_RE.finditer(text))
            if matches:
                line = text[: matches[0].start()].count("\n") + 1
                violations.append((str(path.relative_to(ROOT)), line))
        return violations

    def test_no_inline_torch_imports(self):
        """backends/ files should use ensure_torch(), not inline try/except."""
        violations = self._violating_files()
        if violations:
            msg = "Found inline torch try/except imports (use lazy_import.ensure_torch):\n"
            msg += "\n".join(f"  {f}:{line}" for f, line in violations)
            pytest.fail(msg)


# ─────────────────────────────────────────────────────────────────
# 3. No duplicate try/except import blocks in core/__init__.py
# ─────────────────────────────────────────────────────────────────
_TRY_IMPORT_RE = re.compile(
    r"try\s*:\s*\n\s+(from|import)\s+",
    re.MULTILINE,
)


class TestCoreInitUsesSafeImport:
    """``core/__init__.py`` should use ``safe_import`` for all optional
    imports instead of try/except blocks."""

    def test_no_try_except_imports(self):
        init_path = CORE_DIR / "__init__.py"
        if not init_path.exists():
            pytest.skip("core/__init__.py not found")
        text = init_path.read_text(encoding="utf-8", errors="replace")
        matches = list(_TRY_IMPORT_RE.finditer(text))
        if matches:
            lines = [text[: m.start()].count("\n") + 1 for m in matches]
            pytest.fail(
                f"core/__init__.py has {len(matches)} try/except import blocks "
                f"(lines {lines}). Use safe_import from core.import_utils."
            )


# ─────────────────────────────────────────────────────────────────
# 4. No hand-rolled split_heads / merge_heads in layers/
# ─────────────────────────────────────────────────────────────────
_RESHAPE_HEAD_RE = re.compile(
    r"\.reshape\([^)]*num_heads[^)]*head_dim"
    r"|\.reshape\([^)]*head_dim[^)]*num_heads",
)


class TestNoInlineHeadReshapes:
    """Layer files should use ``attention_utils.split_heads`` /
    ``merge_heads`` instead of manual reshape with num_heads/head_dim."""

    def _violating_files(self) -> List[Tuple[str, int]]:
        violations = []
        for path in _py_files(LAYER_DIR):
            if str(path) in ALLOWED_ATTENTION_RESHAPE_FILES:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            matches = list(_RESHAPE_HEAD_RE.finditer(text))
            if matches:
                line = text[: matches[0].start()].count("\n") + 1
                violations.append((str(path.relative_to(ROOT)), line))
        return violations

    def test_no_inline_head_reshapes(self):
        """layers/ should use attention_utils.split_heads/merge_heads."""
        violations = self._violating_files()
        if violations:
            msg = "Found manual head reshape (use attention_utils):\n"
            msg += "\n".join(f"  {f}:{line}" for f, line in violations)
            pytest.fail(msg)


# ─────────────────────────────────────────────────────────────────
# 5. No duplicate CompositeWrapper-style classes in layers/
# ─────────────────────────────────────────────────────────────────
class TestNoRedundantWrapperClasses:
    """Wrapper classes in ultra_layer_integration.py should use
    ``CompositeWrapper`` via ``functools.partial``, not full class defs."""

    def test_no_duplicate_wrapper_pattern(self):
        target = LAYER_DIR / "ultra_layer_integration.py"
        if not target.exists():
            pytest.skip("ultra_layer_integration.py not found")

        text = target.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(text)
        except SyntaxError as exc:
            pytest.fail(f"ultra_layer_integration.py has a syntax error: {exc}")

        wrapper_classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                name = node.name
                if name.endswith("Wrapper") and name != "CompositeWrapper":
                    wrapper_classes.append(name)

        if wrapper_classes:
            pytest.fail(
                f"Found full class definitions for wrappers that should be "
                f"functools.partial aliases of CompositeWrapper: "
                f"{wrapper_classes}"
            )


# ─────────────────────────────────────────────────────────────────
# 6. Utility modules exist and are importable
# ─────────────────────────────────────────────────────────────────
class TestFactorizationUtilitiesExist:
    """Ensure all centralized utility modules exist and export the
    expected symbols."""

    def test_jax_compat_importable(self):
        from layers.jax_compat import jax, jnp, nn, JAX_AVAILABLE  # noqa: F401

    def test_attention_utils_importable(self):
        from layers.attention_utils import split_heads, merge_heads  # noqa: F401

    def test_import_utils_importable(self):
        from core.import_utils import safe_import, safe_import_module, jax_bundle  # noqa: F401

    def test_lazy_import_importable(self):
        from core.backends.lazy_import import ensure_torch, ensure_jax  # noqa: F401

    def test_field_helpers_importable(self):
        from utils.field_helpers import dict_field, list_field, set_field  # noqa: F401
