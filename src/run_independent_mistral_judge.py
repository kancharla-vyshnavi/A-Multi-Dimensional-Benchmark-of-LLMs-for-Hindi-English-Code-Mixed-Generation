import csv
import json
import re
import time
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_CSV = (
    BASE_DIR
    / "outputs"
    / "controlled_1000"
    / "benchmark_generations_1000.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
    / "independent_judge_1000"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "independent_mistral_judge_1000_results.csv"
)

OUTPUT_JSONL = (
    OUTPUT_DIR
    / "independent_mistral_judge_1000_results.jsonl"
)

CONFIG_JSON = (
    OUTPUT_DIR
    / "independent_mistral_judge_1000_config.json"
)

JUDGE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

MODELS = [
    "Phi-3.5-mini",
    "Qwen2.5-3B",
    "HingGPT",
    "Qwen2.5-7B",
]

SCORE_FIELDS = [
    "fluency",
    "code_mixing_naturalness",
    "hindi_grammar",
    "prompt_adherence",
    "spelling_consistency",
    "overall",
]

BATCH_SIZE = 2
MAX_INPUT_TOKENS = 512
MAX_NEW_TOKENS = 96
SEED = 42

torch.manual_seed(SEED)

# ============================================================
# CHECK GPU
# ============================================================

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU not available.")

print("=" * 80)
print("INDEPENDENT MISTRAL JUDGE - 1000 PROMPTS x 4 MODELS")
print("=" * 80)
print("GPU   :", torch.cuda.get_device_name(0))
print("Input :", INPUT_CSV)
print("Output:", OUTPUT_CSV)
print("Batch :", BATCH_SIZE)
print("=" * 80)

# ============================================================
# LOAD INPUT CSV
# ============================================================

if not INPUT_CSV.exists():
    raise FileNotFoundError(f"Input not found: {INPUT_CSV}")

with open(
    INPUT_CSV,
    "r",
    encoding="utf-8-sig",
    newline=""
) as f:
    rows = list(csv.DictReader(f))

print(f"\nLoaded input rows: {len(rows)}")

if len(rows) != 4000:
    raise ValueError(
        f"Expected 4000 generated rows, found {len(rows)}"
    )

counts = {}

for row in rows:
    model = str(row.get("model", ""))
    counts[model] = counts.get(model, 0) + 1

print("\nMODEL COUNTS:")
for model in MODELS:
    print(f"  {model}: {counts.get(model, 0)}")

for model in MODELS:
    if counts.get(model, 0) != 1000:
        raise ValueError(
            f"{model}: expected 1000, found {counts.get(model, 0)}"
        )

# ============================================================
# LOAD EXISTING RESULTS FOR RESUME
# ============================================================

completed = set()
existing_results = []

if OUTPUT_CSV.exists():
    with open(
        OUTPUT_CSV,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:
        existing_results = list(csv.DictReader(f))

    for row in existing_results:
        try:
            key = (
                int(row["row_index"]),
                str(row["model"]),
            )
            if row.get("overall", "").strip():
                completed.add(key)
        except Exception:
            pass

print(f"\nExisting completed judgments: {len(completed)}")

# ============================================================
# LOAD MISTRAL
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    JUDGE_MODEL,
    use_fast=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "left"

print("Loading Mistral-7B-Instruct-v0.3 in 4-bit...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    JUDGE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)

model.eval()

print("Mistral loaded successfully.")

# ============================================================
# JUDGE PROMPT
# ============================================================

def build_judge_prompt(prompt, response):
    return f"""
You are an independent evaluator of Hindi-English code-mixed text generation.

Evaluate ONLY the model response against the user prompt.

USER PROMPT:
{prompt}

MODEL RESPONSE:
{response}

Score each dimension from 1 to 5.

1. fluency
2. code_mixing_naturalness
3. hindi_grammar
4. prompt_adherence
5. spelling_consistency
6. overall

Scoring:
1 = very poor
2 = poor
3 = acceptable
4 = good
5 = excellent

Focus on the quality of the response itself.
Do not compare models.
Do not reward or penalize the response merely for being short or long.
Consider natural Hindi-English code-mixing appropriate to the prompt.

Return ONLY one valid JSON object.
No explanation.
No markdown.
Do not copy any example.
Values for all six fields MUST be integers from 1 to 5.
Evaluate the actual response before assigning scores.

Required field names:
fluency
code_mixing_naturalness
hindi_grammar
prompt_adherence
spelling_consistency
overall
""".strip()

# ============================================================
# PARSE JUDGE OUTPUT
# ============================================================

def parse_scores(text):
    text = str(text).strip()

    text = re.sub(
        r"```json",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"```",
        "",
        text
    ).strip()

    candidates = []

    try:
        candidates.append(json.loads(text))
    except Exception:
        pass

    match = re.search(
        r"\{.*?\}",
        text,
        flags=re.DOTALL
    )

    if match:
        try:
            candidates.append(
                json.loads(match.group(0))
            )
        except Exception:
            pass

    for obj in candidates:
        if not isinstance(obj, dict):
            continue

        result = {}

        valid = True

        for field in SCORE_FIELDS:
            value = obj.get(field)

            try:
                value = int(float(value))
            except Exception:
                valid = False
                break

            if value < 1 or value > 5:
                valid = False
                break

            result[field] = value

        if valid:
            return result

    return None

# ============================================================
# SAVE
# ============================================================

FIELDNAMES = [
    "row_index",
    "model",
    "prompt",
    "response",
    *SCORE_FIELDS,
    "judge_model",
    "raw_judge_output",
]

results = existing_results.copy()

def save_results():
    ordered = {}

    for row in results:
        try:
            key = (
                int(row["row_index"]),
                str(row["model"]),
            )
        except Exception:
            continue

        ordered[key] = row

    sorted_rows = [
        ordered[key]
        for key in sorted(
            ordered,
            key=lambda x: (x[0], x[1])
        )
    ]

    with open(
        OUTPUT_CSV,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=FIELDNAMES
        )
        writer.writeheader()
        writer.writerows(sorted_rows)

    with open(
        OUTPUT_JSONL,
        "w",
        encoding="utf-8"
    ) as f:
        for row in sorted_rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False
                )
                + "\n"
            )

# ============================================================
# PENDING ROWS
# ============================================================

pending = []

for row in rows:
    try:
        key = (
            int(row["row_index"]),
            str(row["model"]),
        )
    except Exception:
        continue

    if key not in completed:
        pending.append(row)

print(f"\nPending judgments: {len(pending)}")

# ============================================================
# JUDGING
# ============================================================

start_time = time.time()

for start in range(0, len(pending), BATCH_SIZE):

    batch = pending[start:start + BATCH_SIZE]

    prompts = [
        build_judge_prompt(
            str(row["prompt"]),
            str(row["response"])
        )
        for row in batch
    ]

    print(
        f"\n[{start + len(batch)}/{len(pending)}] "
        f"Judging batch of {len(batch)}"
    )

    try:
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_INPUT_TOKENS,
        )

        encoded = {
            k: v.to("cuda")
            for k, v in encoded.items()
        }

        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        input_len = encoded["input_ids"].shape[1]

        decoded = tokenizer.batch_decode(
            generated[:, input_len:],
            skip_special_tokens=True,
        )

        for row, raw_output in zip(batch, decoded):

            scores = parse_scores(raw_output)

            result = {
                "row_index": int(row["row_index"]),
                "model": str(row["model"]),
                "prompt": str(row["prompt"]),
                "response": str(row["response"]),
                "judge_model": JUDGE_MODEL,
                "raw_judge_output": raw_output.strip(),
            }

            for field in SCORE_FIELDS:
                result[field] = (
                    scores[field]
                    if scores is not None
                    else ""
                )

            results.append(result)

            if scores is None:
                print(
                    f"  INVALID JUDGE OUTPUT: "
                    f"{row['model']} / {row['row_index']}"
                )
            else:
                print(
                    f"  OK {row['model']} "
                    f"row={row['row_index']} "
                    f"overall={scores['overall']}"
                )

        save_results()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:

        print(
            f"\nBATCH ERROR: {repr(e)}"
        )

        save_results()

        raise

# ============================================================
# FINAL SUMMARY
# ============================================================

elapsed = time.time() - start_time

save_results()

print("\n" + "=" * 80)
print("MISTRAL JUDGE COMPLETE")
print("=" * 80)

print(f"Total judgments saved: {len(results)}")
print(f"Expected              : 4000")
print(f"Time                  : {elapsed / 60:.2f} minutes")

for model_name in MODELS:

    model_rows = [
        r
        for r in results
        if r.get("model") == model_name
    ]

    valid = [
        r
        for r in model_rows
        if str(r.get("overall", "")).strip()
    ]

    print(
        f"{model_name:20s} "
        f"{len(valid)}/{len(model_rows)} valid"
    )

print("\nCSV   :", OUTPUT_CSV)
print("JSONL :", OUTPUT_JSONL)
print("CONFIG:", CONFIG_JSON)

config = {
    "judge_model": JUDGE_MODEL,
    "input_csv": str(INPUT_CSV),
    "output_csv": str(OUTPUT_CSV),
    "output_jsonl": str(OUTPUT_JSONL),
    "total_input_rows": 4000,
    "responses_per_model": 1000,
    "batch_size": BATCH_SIZE,
    "max_input_tokens": MAX_INPUT_TOKENS,
    "max_new_tokens": MAX_NEW_TOKENS,
    "seed": SEED,
    "scores": SCORE_FIELDS,
}

with open(
    CONFIG_JSON,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        config,
        f,
        indent=2,
        ensure_ascii=False
    )

print("=" * 80)
