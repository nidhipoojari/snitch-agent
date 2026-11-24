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
| 1 | Generate long friendly attacks and normal security emails (Claude Haiku) | Done: 36 attacks, 60 emails |
| 2 | (same run as step 1) | — |
| 3 | Read both new files by hand and delete bad entries | To do |
| 4 | ~~Remove the filler sentence behind the false alarms on our test set~~ | **Skipped** (see below) |
| 5 | Rebuild the dataset: add new rows **without changing the old ones**, so the old test set stays the same and scores are comparable with the mid-term | Done: `data/master_v2.jsonl` |
| 6 | Retrain on Colab with `train_guard.py` | Done: `our_guard_v2` on Drive |
| 7 | Check: `compare_guards.py`, `guard_check.py`, `compare_baselines.py` | In progress |
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

**Result:** 36 attacks (500–700 characters each, 3 to 4 times longer than any v1 attack) and
60 security emails (110–420 characters). All are on topic, and no email talks to an AI.
Each text is about 150 tokens, well under the guard's 512-token window.

One thing to watch: half of the attacks use a short link (`tinyurl.example`, `bit.example`,
`short.example`) and none of the normal emails do, so the guard could learn "short link = attack".
We check for this in step 7.

## Step 4: skipped on purpose

All 17 false alarms on our test set come from one normal sentence:
"Send the invoice to accounts@ourcompany.example, not to me." It was put in the test set on purpose
as a *tricky normal*: a normal text that contains an instruction.

We decided **not** to remove it. Taking a test example out because the guard gets it wrong would make
the scores look better without making the guard better, and the test set would no longer match the
mid-term's. Instead we keep it, report it as a known weakness, and check in step 7 whether the new
security emails (step 1) reduce these false alarms.

## Step 5: the new dataset

**What:** a new option, `--v2`, in `build_master.py`. It builds all the old rows exactly as before,
then **adds** the new rows after them. The new rows use their own random numbers, so they cannot
change anything about the old rows.

```
python build_master.py --raw <folder with BIPIA, InjecAgent, aeslc> --no-deepset --v2
```

**Result:** `data/master_v2.jsonl` has 4,416 rows = the 4,176 old rows + 240 new ones.

- **Old rows:** checked by code to be **identical** to `data/master.jsonl` (text, label, split and
  span). So the old test set is the same as the mid-term's.
- **New rows:** their ids start with `v2_`, so they are easy to pick out.
  - 60 security emails (normal).
  - 180 long-attack rows: each of the 36 attacks is put into 5 different normal texts, the same
    way we did for the v1 attacks. Some of these normal texts are the new security emails, so the
    guard has to find the attack itself and cannot just learn "security email = normal".

| | Train | Val | Test | Total |
|---|---|---|---|---|
| New normal | 40 | 8 | 12 | 60 |
| New attack | 125 | 25 | 30 | 180 |

- **Splits:** the new texts are split 70/15/15 like before. Each attack is in only one split, so the
  test attacks are ones the guard never trained on.
- **Old file kept:** `data/master.jsonl` is not changed. We keep it for the mid-term comparison.

**Files changed or created:**

| File | What happened |
|---|---|
| `build_master.py` | Added the `--v2` option (and an `id_prefix` so new ids start with `v2_`) |
| `data/master_v2.jsonl` | **New**: the dataset for training guard v2 |
| `data/master.jsonl` | **Not touched** |

**In step 7 we report two test scores:** on the old test rows (to compare with the mid-term) and on
the new `v2_` test rows (to see whether long attacks and security emails improved).

## Step 6: retraining

**What:** the same `train_guard.py` and the same settings as the mid-term (DeBERTa-v3-base, 3 epochs),
trained on `master_v2.jsonl` on a free Colab T4 GPU. The output is `our_guard_v2/`, on Drive in
`IS755_Project`, next to the old `our_guard/` (which is not touched).

```
python train_guard.py --data master_v2.jsonl --out our_guard_v2
```

**Scores on the whole v2 test set (old + new test rows together):**

| | Mid-term guard | Guard v2 |
|---|---|---|
| F1 | 0.966 | 0.970 |
| Attacks caught | 356 of 364 | 390 of 394 |
| False alarms | 17 of 372 | 20 of 384 |
| Span F1 (marked the right words) | 0.973 | 0.986 |

This mixes old and new test rows, so it is **not** the fair comparison. Step 7 splits them.

## Step 7: checking (in progress)

New and changed scripts:

| File | What it does |
|---|---|
| `compare_guards.py` | **New.** Compares the two guards on the **old** test rows only (fair comparison with the mid-term), scores guard v2 on the **new** `v2_` rows, and lists every old test row where the two guards disagree. No GPU needed. |
| `guard_check.py` | Added `--guard` and `--out` options, so it can test `our_guard_v2` without replacing the mid-term `guard_check.csv`. |

### Step 7a result: `compare_guards.py` (our own test set)

Guard v2 here is from the **second** training run. The first run's model was lost when Colab
disconnected before it finished saving to Drive. The second run's scores are close to the first's.

**1. Old test rows: the same 736 rows as the mid-term, so this is the fair comparison.**

| | Mid-term guard | Guard v2 |
|---|---|---|
| Attacks caught | 356 of 364 | **360 of 364** |
| Missed | 8 | **4** |
| False alarms | 17 of 372 | 20 of 372 |
| F1 | 0.966 | **0.968** |

- **Fewer misses:** v2 catches 5 BIPIA attacks the old one missed, plus 1 LLM attack. It now misses 2
  BIPIA attacks that the old one caught.
- **False alarms:** the same 17 invoice-sentence ones are still there, plus 3 new ones (2 code
  snippets, 1 Enron email where it marked the single word "creative").

**2. New `v2_` test rows: 30 long attacks caught out of 30, and 0 false alarms on 12 security emails.**
This is encouraging, but these rows were written by the same LLM with the same prompts as the
training rows, so they are easy. The real test is AgentDojo (step 7b), which the guard never saw.

**Note:** every training run comes out a little different, so a change of a few rows (like +4 caught,
+3 false alarms) is about the size of that run-to-run difference.
