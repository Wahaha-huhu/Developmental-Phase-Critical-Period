from __future__ import annotations

from pathlib import Path

p = Path('scripts/analyze_e1_dense_boundary.py')
if not p.exists():
    raise SystemExit(f'Cannot find {p}; run from repository root.')
text = p.read_text()
orig = text

# Common failure: groupby/apply/reset_index on a dataframe that already has step_num,
# causing pandas ValueError: cannot insert step_num, already exists.
# We patch defensively by replacing reset_index() after groupby operations with a safe helper.
helper = r'''

def _safe_reset_index(df):
    """Reset index without failing when an index level name already exists as a column."""
    import pandas as _pd
    if not hasattr(df, "index"):
        return df
    names = [n for n in getattr(df.index, "names", []) if n is not None]
    collisions = [n for n in names if n in getattr(df, "columns", [])]
    if collisions:
        return df.reset_index(drop=True)
    return df.reset_index()
'''

if 'def _safe_reset_index' not in text:
    # Insert after imports or at top.
    insert_at = 0
    lines = text.splitlines()
    for i, line in enumerate(lines[:80]):
        if line.startswith('def ') or line.startswith('class '):
            insert_at = i
            break
    if insert_at == 0:
        text = helper + '\n' + text
    else:
        lines.insert(insert_at, helper)
        text = '\n'.join(lines) + '\n'

# Replace the most dangerous direct reset_index() calls with safe helper.
# This is intentionally broad but limited to the analysis script.
text = text.replace('.reset_index()','.__SAFE_RESET_INDEX_CALL__()')
text = text.replace('.__SAFE_RESET_INDEX_CALL__()','.reset_index()')

# More targeted transformations for common patterns.
patterns = [
    ('.groupby(group_cols).apply(boundary_votes).reset_index()', '._GROUPBY_APPLY_PLACEHOLDER_'),
]
# Instead of attempting AST parsing, patch known problematic idioms line-by-line.
new_lines = []
for line in text.splitlines():
    stripped = line.strip()
    if stripped.endswith('.reset_index()') and ('groupby' in stripped or 'agg(' in stripped or 'apply(' in stripped):
        indent = line[:len(line)-len(line.lstrip())]
        # If it is a one-line expression like x = df.groupby(...).agg(...).reset_index()
        if '=' in line:
            lhs, rhs = line.split('=', 1)
            rhs = rhs.strip()
            rhs = rhs[:-len('.reset_index()')]
            new_lines.append(f"{lhs}= _safe_reset_index({rhs})")
        else:
            expr = line.strip()[:-len('.reset_index()')]
            new_lines.append(f"{indent}_safe_reset_index({expr})")
    else:
        new_lines.append(line)
text = '\n'.join(new_lines) + '\n'

# Add an even more direct fix: if script explicitly computes transitions with step_num in both
# index and columns, ensure duplicate columns are dropped just before reset_index in common variable names.
# This harmless guard de-duplicates columns after CSV load too.
load_guard = """
    # Guard against duplicate columns from previous processing.
    if hasattr(df, 'columns'):
        df = df.loc[:, ~df.columns.duplicated()].copy()
"""
if "df = df.loc[:, ~df.columns.duplicated()].copy()" not in text:
    text = text.replace("df = pd.read_csv(args.metrics)", "df = pd.read_csv(args.metrics)" + load_guard)
    text = text.replace("df = pd.read_csv(metrics_path)", "df = pd.read_csv(metrics_path)" + load_guard)

if text == orig:
    print('No automatic changes made. Showing likely manual fix: replace problematic reset_index() after groupby with reset_index(drop=True) or _safe_reset_index(...).')
else:
    backup = p.with_suffix(p.suffix + '.bak_stepnum')
    backup.write_text(orig)
    p.write_text(text)
    print(f'Patched {p}. Backup written to {backup}.')
