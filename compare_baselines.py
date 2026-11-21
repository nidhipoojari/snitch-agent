"""
compare_baselines.py - compares OUR GUARD with existing detectors on the SAME test texts.

==============================================================================
WHAT IT DOES (in simple words)
==============================================================================
It takes the 736 test texts from master.jsonl (split == "test") and asks each
detector the same question: "is there an attack in this text? yes or no".
Then it scores every detector the same way, so the comparison is fair.

Detectors compared:
  1. ProtectAI DeBERTa v2   existing detector (built on Microsoft's DeBERTa),
                            the one AgentDojo has built in.  No login needed.
  2. Prompt Guard 2 (86M)   existing detector from Meta. Needs a Hugging Face
                            login and Meta's license accepted (see below).
  3. TF-IDF + logistic      a very simple word-counting classifier, trained here
     regression             on our train split. Shows how hard the task is.
  4. Our guard              our trained span guard (the our_guard folder).
                            It counts a text as an attack if it marks any words.

Existing detectors only say yes/no for a whole text, so we compare at that level
(message level). Only our guard can also point to the exact attack words; that
"span" score is in our_guard/test_metrics.json.

Long texts: the detectors can read 512 tokens at a time, so long texts are read
in overlapping windows, and a text counts as an attack if ANY window is flagged.
This is the fair way: no detector misses an attack just because it sits late
in a long email.

==============================================================================
BEFORE RUNNING (Prompt Guard 2 only)
==============================================================================
1. Make a free Hugging Face account and open
   https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-86M
   Click to accept Meta's license and wait for approval (often minutes).
2. Make an access token: https://huggingface.co/settings/tokens  (type "Read").
3. In Colab run:   from huggingface_hub import login; login()   and paste the token.
If you skip this, run with --skip promptguard and the rest still works.

==============================================================================
HOW TO RUN (Colab, with the GPU on; files in your IS755_Project folder)
==============================================================================
    pip install transformers accelerate sentencepiece protobuf scikit-learn
    python compare_baselines.py --data master.jsonl --guard our_guard

Quick check that it runs (60 test texts):
    python compare_baselines.py --data master.jsonl --guard our_guard --limit 60

Output (folder --out, default comparison/):
    comparison_table.csv         one row per detector: precision, recall, F1, false alarms
    comparison_by_source.csv     the same per source (bipia, enron, injecagent, ...)
    comparison_predictions.csv   every test text with every detector's yes/no and score
    comparison_summary.json      all numbers + paired tests (ours vs each detector)
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DETECTORS = {
    "protectai": ("ProtectAI DeBERTa v2", "protectai/deberta-v3-base-prompt-injection-v2"),
    "promptguard": ("Prompt Guard 2 (86M)", "meta-llama/Llama-Prompt-Guard-2-86M"),
}


# ============================================================================ existing detectors
def attack_label_id(model):
    """Find which output of the model means 'attack'.
    ProtectAI calls it 'INJECTION'; Prompt Guard 2 calls it 'LABEL_1' (malicious)."""
    labels = {i: str(l).upper() for i, l in model.config.id2label.items()}
    for i, l in labels.items():
        if any(w in l for w in ("INJECTION", "MALICIOUS", "JAILBREAK", "UNSAFE")):
            return i
    if len(labels) == 2:
        return 1                                  # LABEL_0 = benign, LABEL_1 = attack
    raise SystemExit(f"Can't tell which label means 'attack': {labels}")


@torch.no_grad()
def score_detector(model_name, texts, batch_size=32, max_len=512, stride=64, device=None):
    """Attack score for each text (0 to 1). Long texts: highest score over all windows."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).float().to(device).eval()
    attack_id = attack_label_id(model)
    print(f"  labels: {model.config.id2label}  -> 'attack' = {model.config.id2label[attack_id]}")

    enc = tok(texts, truncation=True, max_length=max_len, stride=stride,
              return_overflowing_tokens=True)
    owner = enc["overflow_to_sample_mapping"]
    scores = np.zeros(len(texts), dtype=np.float32)
    n = len(enc["input_ids"])
    for start in range(0, n, batch_size):
        idx = list(range(start, min(start + batch_size, n)))
        batch = tok.pad({"input_ids": [enc["input_ids"][i] for i in idx],
                         "attention_mask": [enc["attention_mask"][i] for i in idx]},
                        return_tensors="pt").to(device)
        probs = torch.softmax(model(**batch).logits.float(), -1)[:, attack_id].cpu().numpy()
        for row, i in enumerate(idx):
            doc = owner[i]
            scores[doc] = max(scores[doc], probs[row])
    print(f"  {len(texts)} texts read as {n} windows")
    return scores


# ============================================================================ simple baseline
def score_tfidf(train_df, texts):
    """Word-counting classifier trained on our train split."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(vec.fit_transform(train_df["text"]), train_df["label"])
    return clf.predict_proba(vec.transform(texts))[:, 1]


# ============================================================================ our guard
def score_our_guard(guard_dir, texts, batch_size=32, threshold=0.5):
    """Our guard: the highest attack score over all characters of the text."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_guard import Guard
    guard = Guard(guard_dir, threshold=threshold)
    spans, max_scores = guard.find_attacks_many(texts, batch_size)
    # a text is flagged when the guard marks at least one span (same rule as train_guard.py)
    flagged = np.array([1 if s else 0 for s in spans])
    return np.array(max_scores, dtype=np.float32), flagged


# ============================================================================ scoring
def scores_for(labels, preds):
    labels, preds = np.asarray(labels), np.asarray(preds)
    tp = int(((labels == 1) & (preds == 1)).sum()); fp = int(((labels == 0) & (preds == 1)).sum())
    fn = int(((labels == 1) & (preds == 0)).sum()); tn = int(((labels == 0) & (preds == 0)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4),
            "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0,
            "false_alarm_rate": round(fp / (fp + tn), 4) if fp + tn else None,
            "accuracy": round((tp + tn) / len(labels), 4) if len(labels) else None,
            "attacks_caught": f"{tp}/{tp + fn}", "false_alarms": f"{fp}/{fp + tn}"}


def mcnemar(labels, ours, theirs):
    """Paired test on the same texts: is our guard right more often than the other detector?
    Counts texts where exactly one of the two is right; exact two-sided binomial test."""
    labels = np.asarray(labels)
    ours_right, theirs_right = (np.asarray(ours) == labels), (np.asarray(theirs) == labels)
    b = int((ours_right & ~theirs_right).sum())       # only ours right
    c = int((~ours_right & theirs_right).sum())       # only theirs right
    n = b + c
    if n == 0:
        return {"only_ours_right": b, "only_theirs_right": c, "p_value": 1.0}
    k = min(b, c)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"only_ours_right": b, "only_theirs_right": c, "p_value": float(f"{p:.3g}")}


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser(description="Compare our guard with existing detectors on the test split.")
    ap.add_argument("--data", default="data/master.jsonl")
    ap.add_argument("--guard", default="our_guard", help="folder with our trained guard")
    ap.add_argument("--out", default="comparison")
    ap.add_argument("--threshold", type=float, default=0.5, help="score above which a text counts as attack")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--skip", nargs="*", default=[], choices=["protectai", "promptguard", "tfidf", "ours"],
                    help="detectors to leave out, e.g. --skip promptguard")
    ap.add_argument("--protectai-model", default=DETECTORS["protectai"][1])
    ap.add_argument("--promptguard-model", default=DETECTORS["promptguard"][1])
    ap.add_argument("--limit", type=int, default=None, help="only this many test texts (quick check)")
    args = ap.parse_args()

    df = pd.read_json(args.data, lines=True)
    train, test = df[df.split == "train"], df[df.split == "test"].reset_index(drop=True)
    if args.limit:
        test = test.sample(min(args.limit, len(test)), random_state=0).reset_index(drop=True)
    texts, labels = test["text"].tolist(), test["label"].to_numpy()
    print(f"Test texts: {len(test)} ({int(labels.sum())} attacks, {int((labels == 0).sum())} normal)")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none (slow)'}\n")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    preds = test[["id", "source", "content_type", "label"]].copy()
    runs = []                                            # (key, display name)

    for key, model_arg in (("protectai", args.protectai_model), ("promptguard", args.promptguard_model)):
        if key in args.skip:
            continue
        name = DETECTORS[key][0]
        print(f"Running {name} ({model_arg}) ...")
        t0 = time.time()
        try:
            s = score_detector(model_arg, texts, args.batch)
        except Exception as e:
            msg = str(e).lower()
            hint = (" -> accept Meta's license on the model page and log in to Hugging Face "
                    "(see the top of this script), or add --skip promptguard"
                    if key == "promptguard" and any(w in msg for w in ("gated", "401", "403", "access", "token"))
                    else "")
            print(f"  [skip] {name} failed: {type(e).__name__}: {str(e)[:200]}{hint}\n")
            continue
        preds[f"{key}_score"] = s.round(4)
        preds[f"{key}_pred"] = (s >= args.threshold).astype(int)
        runs.append((key, name))
        print(f"  done in {time.time() - t0:.0f}s\n")

    if "tfidf" not in args.skip:
        print("Running TF-IDF + logistic regression (trained on our train split) ...")
        s = score_tfidf(train, texts)
        preds["tfidf_score"] = s.round(4)
        preds["tfidf_pred"] = (s >= 0.5).astype(int)
        runs.append(("tfidf", "TF-IDF + logistic regression"))
        print("  done\n")

    if "ours" not in args.skip:
        print(f"Running our guard ({args.guard}) ...")
        s, flagged = score_our_guard(args.guard, texts, args.batch, args.threshold)
        preds["ours_score"] = s.round(4)
        preds["ours_pred"] = flagged
        runs.append(("ours", "Our guard (span-level)"))
        print("  done\n")

    # ---- results table
    table = pd.DataFrame([{"detector": name, **scores_for(labels, preds[f"{key}_pred"])} for key, name in runs])
    by_source = []
    for key, name in runs:
        for src, g in preds.groupby("source"):
            by_source.append({"detector": name, "source": src, "n": len(g), **scores_for(g.label, g[f"{key}_pred"])})
    by_source = pd.DataFrame(by_source)

    tests = {}
    if any(k == "ours" for k, _ in runs):
        for key, name in runs:
            if key != "ours":
                tests[f"ours vs {name}"] = mcnemar(labels, preds["ours_pred"], preds[f"{key}_pred"])

    print("=============== MESSAGE-LEVEL COMPARISON (same test texts) ===============")
    print(table[["detector", "precision", "recall", "f1", "false_alarm_rate", "attacks_caught", "false_alarms"]]
          .to_string(index=False))
    print("\nF1 by source (enron has no attacks, so its F1 is always 0; look at false alarms there):")
    print(by_source.pivot(index="source", columns="detector", values="f1").round(3).to_string())
    print("\nFalse-alarm rate by source (share of NORMAL texts wrongly flagged):")
    print(by_source.pivot(index="source", columns="detector", values="false_alarm_rate").round(3).to_string())
    if tests:
        print("\nPaired test on the same texts (McNemar): p < 0.05 means the difference is not just luck")
        for k, v in tests.items():
            print(f"  {k:45s} only ours right: {v['only_ours_right']:3d}   "
                  f"only theirs right: {v['only_theirs_right']:3d}   p = {v['p_value']}")

    table.to_csv(out / "comparison_table.csv", index=False)
    by_source.to_csv(out / "comparison_by_source.csv", index=False)
    preds.to_csv(out / "comparison_predictions.csv", index=False)
    json.dump({"n_test": len(test), "threshold": args.threshold, "table": table.to_dict(orient="records"),
               "by_source": by_source.to_dict(orient="records"), "paired_tests": tests},
              open(out / "comparison_summary.json", "w"), indent=2)
    print(f"\nSaved comparison_table.csv, comparison_by_source.csv, comparison_predictions.csv, "
          f"comparison_summary.json in {out}/")


if __name__ == "__main__":
    main()
