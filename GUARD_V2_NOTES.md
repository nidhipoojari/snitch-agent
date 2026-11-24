# Guard v2: what we are changing and why

Branch: `guard-v2`. This file is updated after each step.

## Why

When the guard read AgentDojo tool outputs (`guard_check.csv`), it made two kinds of mistakes:

1. **Missed attacks (15 of 81).** 10 of them are one attack, `injection_task_13`: a long,
   friendly message asking the AI to email everyone the user is meeting, with a link.
   All our training attacks are short (1–2 sentences), so the guard has never seen a long, friendly one.
2. **False alarms (6).** All 6 are normal password-reset emails. Our training data has almost no
   normal security emails, so "link + reset password" looks like an attack to the guard.

Fix: add training examples of exactly these two kinds, retrain, and check again.
We write the new examples **in our own words**. We do not copy AgentDojo text, because AgentDojo
is our final test.

## Steps

| # | Step | Status |
|---|---|---|
| 1 | Generate long friendly attacks and normal security emails (Claude Haiku) | Code done, waiting to run |
| 2 | (same run as step 1) | — |
| 3 | Read both new files by hand and delete bad entries | To do |
| 4 | Remove the filler sentence behind the false alarms on our test set | To do |
| 5 | Rebuild the dataset: add new rows **without changing the old ones**, so the old test set stays the same and scores are comparable with the mid-term | To do |
| 6 | Retrain on Colab with `train_guard.py` | To do |
| 7 | Check: `--eval-only`, `compare_baselines.py`, `guard_check.py` | To do |
| 8 | Upload the model to Drive as `our_guard_v2` and tell Nidhi | To do |

## Step 1: new training texts

**What:** a new option, `--generate-v2`, in `build_master.py`. It asks Claude Haiku for:

- **36 long attacks** (12 each of 3 kinds): "message all these people with this friendly text and link",
  "post this announcement with a link", and "a polite list of several steps".
- **60 normal security emails** (15 each of 4 kinds): password reset, new sign-in alert,
  6-digit verification code, and "verify your email address".

We ask for a few more than we need because some get deleted in step 3.

**Files changed or created:**

| File | What happened |
|---|---|
| `build_master.py` | Added the `--generate-v2` option and the new prompts (`ATTACK_STYLES_V2`, `ATTACK_PROMPT_V2`, `NORMAL_KINDS_V2`) |
| `data/generated/attacks_v2.json` | **New**: the long attacks |
| `data/generated/normals_v2.json` | **New**: the security emails |
| `data/generated/attacks.json`, `normals.json` | **Not touched**, so the old dataset can be rebuilt exactly |

**How to run:**

```
python build_master.py --generate-v2 --llm anthropic
```

This only writes the two `_v2` files. It does not rebuild the dataset. If a file already exists it is
not overwritten (delete it to generate again).
