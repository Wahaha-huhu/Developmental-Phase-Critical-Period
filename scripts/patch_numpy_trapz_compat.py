#!/usr/bin/env python3
"""Patch repository Python files for NumPy 2.x compatibility.

NumPy 2.x removed np.trapz. This script replaces direct np.trapz calls with
np.trapezoid calls. It also adds a small compatibility alias to files that use
np.trapezoid if needed is not necessary, because NumPy 2.x supports trapezoid.

Run from project root:
    python scripts/patch_numpy_trapz_compat.py
"""
from pathlib import Path

ROOT = Path.cwd()
TARGET_DIRS = [ROOT / "src", ROOT / "scripts"]

changed = []
for base in TARGET_DIRS:
    if not base.exists():
        continue
    for path in base.rglob("*.py"):
        text = path.read_text()
        new = text.replace("np.trapz(", "np.trapezoid(")
        # Also handle numpy.trapz if used explicitly.
        new = new.replace("numpy.trapz(", "numpy.trapezoid(")
        if new != text:
            path.write_text(new)
            changed.append(str(path.relative_to(ROOT)))

if changed:
    print("Patched NumPy trapz compatibility in:")
    for item in changed:
        print("  -", item)
else:
    print("No np.trapz calls found under src/ or scripts/.")
