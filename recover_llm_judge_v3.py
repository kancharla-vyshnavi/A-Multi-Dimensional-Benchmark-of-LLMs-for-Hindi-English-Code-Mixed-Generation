import os
import re
import json
import ast
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# =========================
# CONFIG
# =========================
JUDGE_MODEL = "microsoft/Phi-3.5-mini-instruct"

INPUT_CSV = r".\outputs\llm_judge\hinglish_bench_llm_judge_results.csv"
OUTPUT_CSV = r".\outputs\llm_judge\hinglish_bench_llm_judge_results_recovered_v3.csv"

MAX_NEW_TOKENS = 80

CRITERIA = [
    "fluency",
    "code_mixing_naturalness",
    "hindi_grammar",
    "prompt_adherence",
    "spelling_consistency",
    "overall_quality",
]


# =========================
# LOAD DATA
# =========================
df = pd.read_csv(INPUT_CSV)

score_cols = CRITERIA

def is_valid_row(row):
    for c in score_cols:
        try:
            x = int(row[c])
            if x < 1 or x > 5:
                return False
        except:
            return False
    return True


invalid_idx = [
    i for i, row in df.iterrows()
    if not is_valid_row(row)
]

print(f"Total rows: {len(df)}")
print(f"Invalid rows to recover: {len(invalid_idx)}")

if not invalid_idx:
    print("Nothing to recover.")
    raise SystemExit


# =========================
# LOAD JUDGE
# =========================
print("\nLoading judge model...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(
    JUDGE_MODEL,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    JUDGE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
    attn_implementation="eager",
)

model.eval()
model.config.use_cache = False

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token


# =========================
# PARSER
# =========================
def parse_scores(text):
    if not text:
        return None

    text = text.strip()

    # ---- JSON objects ----
    json_candidates = re.findall(r"\{.*?\}", text, flags=re.S)

    for candidate in json_candidates:
        # remove markdown fences
        candidate = candidate.replace("```json", "").replace("```", "").strip()

        # fix stray standalone numbers:
        # "hindi_grammar": 2,
        # 0,
        # "prompt_adherence": 1
        candidate = re.sub(
            r",\s*\d+\s*,",
            ",",
            candidate
        )

        try:
            obj = json.loads(candidate)

            values = []
            for c in CRITERIA:
                if c not in obj:
                    break
                values.append(int(obj[c]))
            else:
                if all(1 <= x <= 5 for x in values):
                    return dict(zip(CRITERIA, values))

        except Exception:
            pass

    # ---- Python dict fallback ----
    for candidate in re.findall(r"\{.*?\}", text, flags=re.S):
        try:
            obj = ast.literal_eval(candidate)

            values = []
            for c in CRITERIA:
                if c not in obj:
                    break
                values.append(int(obj[c]))
            else:
                if all(1 <= x <= 5 for x in values):
                    return dict(zip(CRITERIA, values))

        except Exception:
            pass

    # ---- extract six integer scores directly ----
    nums = re.findall(r"\b[1-5]\b", text)

    if len(nums) >= 6:
        vals = list(map(int, nums[-6:]))

        if len(vals) == 6 and all(1 <= x <= 5 for x in vals):
            return dict(zip(CRITERIA, vals))

    return None


# =========================
# JUDGE PROMPT
# =========================
def make_prompt(response):
    return f"""You are evaluating a Hinglish response.

Score each criterion from 1 to 5.

Criteria:
- fluency
- code_mixing_naturalness
- hindi_grammar
- prompt_adherence
- spelling_consistency
- overall_quality

Response:
{response}

Return ONLY one JSON object with exactly these six integer fields.
No explanation.

Example:
{{"fluency":3,"code_mixing_naturalness":3,"hindi_grammar":3,"prompt_adherence":3,"spelling_consistency":3,"overall_quality":3}}
"""


# =========================
# GENERATE
# =========================
def run_judge(response):
    messages = [
        {
            "role": "user",
            "content": make_prompt(str(response))
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=768
    )

    inputs = {
        k: v.to("cuda:0")
        for k, v in inputs.items()
    }

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    new_tokens = output[0][inputs["input_ids"].shape[1]:]

    raw = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True
    ).strip()

    return raw


# =========================
# RECOVERY LOOP
# =========================
recovered = 0
failed = 0

for n, idx in enumerate(invalid_idx, start=1):

    row = df.loc[idx]

    model_name = row.get("model", "")
    sample_id = row.get("sample_id", "")

    response = row.get("response", "")

    print(
        f"\n[{n}/{len(invalid_idx)}] "
        f"{model_name} - sample {sample_id}"
    )

    raw = run_judge(response)

    scores = parse_scores(raw)

    # second attempt with a stricter prompt
    if scores is None:

        print("   Retry...")

        retry_prompt = f"""Give ONLY six numbers separated by spaces.

Order:
fluency
code_mixing_naturalness
hindi_grammar
prompt_adherence
spelling_consistency
overall_quality

Each number must be between 1 and 5.

Response:
{response}
"""

        messages = [
            {
                "role": "user",
                "content": retry_prompt
            }
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=768
        )

        inputs = {
            k: v.to("cuda:0")
            for k, v in inputs.items()
        }

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=False,
                use_cache=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        new_tokens = output[0][inputs["input_ids"].shape[1]:]

        raw2 = tokenizer.decode(
            new_tokens,
            skip_special_tokens=True
        ).strip()

        scores = parse_scores(raw2)

        raw = raw2

    if scores is not None:

        for c in CRITERIA:
            df.loc[idx, c] = scores[c]

        df.loc[idx, "judge_raw_output"] = raw

        recovered += 1

        print("   RECOVERED:", scores)

    else:

        failed += 1

        print("   FAILED")
        print("   RAW:", repr(raw[:300]))

    # save after EVERY row
    df.to_csv(
        OUTPUT_CSV,
        index=False
    )


# =========================
# FINAL CHECK
# =========================
valid_final = sum(
    is_valid_row(row)
    for _, row in df.iterrows()
)

invalid_final = len(df) - valid_final

print("\n" + "=" * 60)
print("RECOVERY COMPLETE")
print("=" * 60)

print("Recovered this run :", recovered)
print("Failed this run    :", failed)
print("Final valid rows   :", valid_final)
print("Final invalid rows :", invalid_final)

print("\nSaved:")
print(OUTPUT_CSV)

if invalid_final == 0:
    print("\nSUCCESS: 136/136 rows are valid.")
else:
    print(
        f"\nStill invalid: {invalid_final} rows."
    )