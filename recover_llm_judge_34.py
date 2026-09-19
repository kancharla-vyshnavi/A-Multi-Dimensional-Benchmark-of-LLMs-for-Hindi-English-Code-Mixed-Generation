import pandas as pd
import re
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ============================================================
# PATHS
# ============================================================

INPUT_CSV = r".\outputs\llm_judge\hinglish_bench_llm_judge_results_fast.csv"
OUTPUT_CSV = r".\outputs\llm_judge\hinglish_bench_llm_judge_results_final.csv"

JUDGE_MODEL = "microsoft/Phi-3.5-mini-instruct"

COLS = [
    "fluency",
    "code_mixing_naturalness",
    "hindi_grammar",
    "prompt_adherence",
    "spelling_consistency",
    "overall",
]

MAX_NEW_TOKENS = 35


# ============================================================
# VALIDATION
# ============================================================

def is_valid(row):
    try:
        values = [int(row[c]) for c in COLS]
        return all(1 <= x <= 5 for x in values)
    except Exception:
        return False


# ============================================================
# LOAD EXISTING 102 VALID + 34 INVALID
# ============================================================

df = pd.read_csv(INPUT_CSV)

invalid_indices = [
    i for i, row in df.iterrows()
    if not is_valid(row)
]

print("=" * 60)
print("LLM JUDGE FINAL RECOVERY")
print("=" * 60)
print(f"Total rows     : {len(df)}")
print(f"Already valid  : {len(df) - len(invalid_indices)}")
print(f"Remaining      : {len(invalid_indices)}")
print("=" * 60)


if len(invalid_indices) == 0:
    df.to_csv(OUTPUT_CSV, index=False)
    print("ALL 136 ROWS ARE ALREADY VALID.")
    print("Saved:", OUTPUT_CSV)
    raise SystemExit


# ============================================================
# LOAD PHI JUDGE
# ============================================================

print("\nLoading Phi-3.5-mini judge...")

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

print("Judge loaded.")
print("=" * 60)


# ============================================================
# PARSE SIX SCORES
# ============================================================

def parse_scores(raw):

    # --------------------------------------------------------
    # 1. Try JSON
    # --------------------------------------------------------

    objects = re.findall(r"\{.*?\}", raw, flags=re.S)

    for obj_text in objects:

        # Remove markdown fences
        obj_text = obj_text.replace("```json", "")
        obj_text = obj_text.replace("```", "")
        obj_text = obj_text.strip()

        # Remove stray standalone number lines
        obj_text = re.sub(
            r",\s*\d+\s*,",
            ",",
            obj_text
        )

        try:
            obj = json.loads(obj_text)

            values = [
                int(obj[c])
                for c in COLS
            ]

            if all(1 <= x <= 5 for x in values):
                return values

        except Exception:
            pass

    # --------------------------------------------------------
    # 2. Extract last six valid numbers
    # --------------------------------------------------------

    nums = re.findall(r"\b[1-5]\b", raw)

    if len(nums) >= 6:

        values = [
            int(x)
            for x in nums[-6:]
        ]

        if all(1 <= x <= 5 for x in values):
            return values

    return None


# ============================================================
# JUDGE ONE RESPONSE
# ============================================================

def run_judge(response):

    prompt = f"""
Evaluate the following Hinglish response.

Give SIX scores from 1 to 5.

Order:
1. Fluency
2. Code-mixing naturalness
3. Hindi grammar
4. Prompt adherence
5. Spelling consistency
6. Overall quality

IMPORTANT:
Return ONLY six numbers separated by spaces.

Example:
3 4 3 4 3 4

Response:
{response}
"""

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=768
    )

    inputs = {
        key: value.to("cuda:0")
        for key, value in inputs.items()
    }

    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = output[
        0,
        inputs["input_ids"].shape[1]:
    ]

    raw = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    scores = parse_scores(raw)

    return raw, scores


# ============================================================
# RECOVER ONLY 34 INVALID ROWS
# ============================================================

recovered = 0
failed = 0

for count, idx in enumerate(invalid_indices, start=1):

    row = df.loc[idx]

    model_name = str(row["model"])
    sample_id = str(row["sample_id"])
    response = str(row["response"])

    print(
        f"\n[{count}/{len(invalid_indices)}] "
        f"{model_name} - sample {sample_id}"
    )

    raw, scores = run_judge(response)

    # --------------------------------------------------------
    # Successful
    # --------------------------------------------------------

    if scores is not None:

        for column, score in zip(COLS, scores):
            df.loc[idx, column] = score

        df.loc[idx, "judge_raw_output"] = raw

        recovered += 1

        print("   RECOVERED:", scores)

    # --------------------------------------------------------
    # Failed -> one short retry
    # --------------------------------------------------------

    else:

        print("   Retry...")

        retry_prompt = f"""
Score this Hinglish response.

Return ONLY six integers separated by spaces.
Each must be 1, 2, 3, 4, or 5.

Order:
fluency code_mixing_naturalness hindi_grammar prompt_adherence spelling_consistency overall

Response:
{response}
"""

        messages = [
            {
                "role": "user",
                "content": retry_prompt
            }
        ]

        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=768
        )

        inputs = {
            key: value.to("cuda:0")
            for key, value in inputs.items()
        }

        with torch.no_grad():

            output = model.generate(
                **inputs,
                max_new_tokens=25,
                do_sample=False,
                use_cache=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated_tokens = output[
            0,
            inputs["input_ids"].shape[1]:
        ]

        retry_raw = tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        ).strip()

        retry_scores = parse_scores(retry_raw)

        if retry_scores is not None:

            for column, score in zip(COLS, retry_scores):
                df.loc[idx, column] = score

            df.loc[idx, "judge_raw_output"] = retry_raw

            recovered += 1

            print("   RETRY RECOVERED:", retry_scores)

        else:

            failed += 1

            print("   FAILED")
            print("   RAW:", repr(retry_raw[:200]))

    # --------------------------------------------------------
    # SAVE AFTER EVERY ROW
    # --------------------------------------------------------

    df.to_csv(
        OUTPUT_CSV,
        index=False
    )


# ============================================================
# FINAL CHECK
# ============================================================

final_valid = sum(
    is_valid(row)
    for _, row in df.iterrows()
)

final_invalid = len(df) - final_valid

print("\n")
print("=" * 60)
print("FINAL RECOVERY RESULT")
print("=" * 60)
print("Recovered :", recovered)
print("Failed    :", failed)
print("Valid     :", final_valid)
print("Invalid   :", final_invalid)
print("=" * 60)
print("Saved to:")
print(OUTPUT_CSV)
print("=" * 60)

if final_invalid == 0:
    print("\nSUCCESS: 136/136 VALID.")
else:
    print(
        f"\nRemaining invalid rows: {final_invalid}"
    )