#!/usr/bin/env python3
"""Patch E3 v3 runner so conflicting-fact wrong values are fixed per fact.

Why: for nested poison budgets [4,16,64,256], the first 4 attacked facts should use
exactly the same wrong values in k=4, k=16, k=64, and k=256. Otherwise budget changes
also change the target contradiction, which adds avoidable variance.
"""
from pathlib import Path

path = Path("scripts/run_e3_factual_cell_v3.py")
if not path.exists():
    raise SystemExit(f"Could not find {path}; run from repository root.")

text = path.read_text()
start_marker = "    # Poison budgets: same trained entities/relations, wrong values."
end_marker = "    # Sanity checks for the prior bug: probes must query taught facts, controls must not."
start = text.find(start_marker)
end = text.find(end_marker)
if start == -1 or end == -1 or end <= start:
    raise SystemExit("Could not locate poison-budget block to patch. Maybe already patched or code changed.")

new_block = '''    # Poison budgets: same trained entities/relations, fixed wrong values.
    # Budget k means k contradicted facts. Each contradicted fact is rendered through the same templates.
    # Important final-run control: wrong values are precomputed once per fact, so nested budgets are
    # truly nested. The first 4 attacked facts use the same wrong values in k=4, k=16, k=64, etc.
    budgets = [int(x) for x in config.get("degradation", {}).get("poison_budgets", [])]
    shuffled_facts = facts.copy()
    rng.shuffle(shuffled_facts)

    wrong_value_by_fact_id: Dict[int, str] = {}
    for f in shuffled_facts:
        wrong_pool = [v for v in all_values if v != f.value]
        wrong_value_by_fact_id[f.fact_id] = rng.choice(wrong_pool)

    poison_fact_ids_by_budget: Dict[int, List[int]] = {}
    poison_by_budget: Dict[int, List[str]] = {}
    for k in budgets:
        attacked = shuffled_facts[: min(k, len(shuffled_facts))]
        poison_fact_ids_by_budget[k] = [f.fact_id for f in attacked]
        texts: List[str] = []
        for f in attacked:
            wrong = wrong_value_by_fact_id[f.fact_id]
            if wrong == f.value:
                raise AssertionError("BUG: adversarial wrong value equals correct value")
            texts.extend(render_fact_texts(f, train_templates, value_override=wrong))
        poison_by_budget[k] = texts

'''
text = text[:start] + new_block + text[end:]

# Add audit fields if the known audit block exists.
audit_marker = '        "example_control": controls[0] if controls else None,\n'
if audit_marker in text and '"poison_budgets": budgets,' not in text:
    insert = '''        "poison_budgets": budgets,
        "poison_budget_unit": "facts",
        "poison_templates_per_fact": len(train_templates),
        "poison_budget_fact_ids": poison_fact_ids_by_budget,
        "example_wrong_value_by_fact_id": {str(k): v for k, v in list(wrong_value_by_fact_id.items())[:5]},
'''
    text = text.replace(audit_marker, audit_marker + insert)

backup = path.with_suffix(path.suffix + ".bak_adversarial_values")
if not backup.exists():
    backup.write_text(path.read_text())
path.write_text(text)
print(f"Patched {path}")
print(f"Backup at {backup}")
