"""
Compare the mid-term guard (our_guard) with guard v2 (our_guard_v2) on the test set.

Both folders need test_predictions.jsonl (written by train_guard.py). No GPU or model needed.

  1. OLD test rows: the same 736 rows the mid-term guard was scored on -> a fair comparison.
  2. NEW test rows: the v2_ rows (long attacks and security emails) -> did the fix work?
  3. Every old test row where the two guards disagree, so we can see what changed.

Run:  python compare_guards.py
      python compare_guards.py --old our_guard --new our_guard_v2
"""

import argparse
from pathlib import Path

import pandas as pd


def scores(p):
    """Message-level scores for one group of prediction rows."""
    atk, nor = p[p.label == 1], p[p.label == 0]
    tp, fp = int(atk.predicted.sum()), int(nor.predicted.sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / len(atk) if len(atk) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"texts": len(p), "attacks": len(atk), "caught": tp, "missed": len(atk) - tp,
            "normals": len(nor), "false alarms": fp, "F1": round(f1, 3)}


def table(named_scores):
    df = pd.DataFrame(named_scores).T
    return df.astype({c: int for c in df.columns if c != "F1"}).to_string()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default="our_guard", help="folder of the mid-term guard")
    ap.add_argument("--new", default="our_guard_v2", help="folder of guard v2")
    args = ap.parse_args()

    old = pd.read_json(Path(args.old) / "test_predictions.jsonl", lines=True)
    new = pd.read_json(Path(args.new) / "test_predictions.jsonl", lines=True)
    is_v2 = new.id.str.startswith("v2_")
    new_old_rows, new_v2_rows = new[~is_v2], new[is_v2]

    missing = set(old.id) - set(new_old_rows.id)
    if missing:
        print(f"WARNING: {len(missing)} old test rows are missing from {args.new}, e.g. {sorted(missing)[:3]}")

    print("\n1. OLD test rows (same rows as the mid-term: fair comparison)")
    print(table({"mid-term guard": scores(old), "guard v2": scores(new_old_rows)}))

    print("\n2. NEW test rows (v2_: long attacks and security emails)")
    print(table({"guard v2": scores(new_v2_rows)}))
    for label, name in ((1, "missed long attacks"), (0, "false alarms on security emails")):
        bad = new_v2_rows[(new_v2_rows.label == label) & (new_v2_rows.predicted != label)]
        print(f"   {name}: {len(bad)}" + (f"  {bad.id.tolist()}" if len(bad) else ""))

    both = old.merge(new_old_rows, on="id", suffixes=("_old", "_new"))
    changed = both[both.predicted_old != both.predicted_new]
    print(f"\n3. OLD test rows where the two guards disagree: {len(changed)}")
    for r in changed.itertuples():
        kind = "attack" if r.label_old == 1 else "normal"
        verdict = "now CORRECT" if r.predicted_new == r.label_old else "now WRONG"
        marked = " | ".join(t.strip()[:100] for t in r.pred_text_new) if r.predicted_new else ""
        print(f"   {verdict:12} {kind:7} {r.id}" + (f"\n                marked: {marked}" if marked else ""))


if __name__ == "__main__":
    main()
