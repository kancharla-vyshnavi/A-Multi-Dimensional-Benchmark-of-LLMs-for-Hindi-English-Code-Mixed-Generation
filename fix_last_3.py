import os
import json
import re
import pandas as pd
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)


# ============================================================
# PATHS
# ============================================================

MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"

FINAL_JSONL = r".\outputs\pairwise_judge\pairwise_judge_results_final.jsonl"
FINAL_CSV = r".\outputs\pairwise_judge\pairwise_judge_results_final.csv"


# ============================================================
# TARGET ROWS
# ============================================================

TARGETS = {
    ("Phi-3.5-mini", "Qwen2.5-3B", "16"),
    ("Phi-3.5-mini", "Qwen2.5-3B", "17"),
    ("Phi-3.5-mini", "Qwen2.5-7B", "15"),
}


# ============================================================
# PARSER
# ============================================================

def parse_judgment(text):

    if text is None:
        return None

    text = str(text).strip().upper()

    # Example:
    # A B B T A A

    tokens = re.findall(
        r"\b(?:A|B|T|TIE)\b",
        text
    )

    if len(tokens) >= 6:

        tokens = tokens[-6:]

        result = []

        for token in tokens:

            if token == "TIE":
                token = "T"

            result.append(token)

        if all(
            x in {"A", "B", "T"}
            for x in result
        ):

            return " ".join(result)

    # Example:
    # ABBTAA

    compact = re.sub(
        r"[^ABT]",
        "",
        text
    )

    if len(compact) >= 6:

        candidate = compact[-6:]

        if all(
            x in {"A", "B", "T"}
            for x in candidate
        ):

            return " ".join(
                list(candidate)
            )

    return None


# ============================================================
# LOAD FINAL RESULTS
# ============================================================

with open(
    FINAL_JSONL,
    "r",
    encoding="utf-8"
) as f:

    rows = [
        json.loads(line)
        for line in f
        if line.strip()
    ]


print("=" * 70)
print("FIXING FINAL 3 PAIRWISE JUDGMENTS")
print("=" * 70)

print(
    f"\nLoaded rows: {len(rows)}"
)


# ============================================================
# FIND TARGET ROWS
# ============================================================

target_rows = []

for index, row in enumerate(rows):

    key = (
        str(row.get("pair_model_1", "")).strip(),
        str(row.get("pair_model_2", "")).strip(),
        str(row.get("sample_id", "")).strip()
    )

    if key in TARGETS:

        target_rows.append(
            (index, row)
        )


print(
    f"Target rows found: {len(target_rows)}"
)

for index, row in target_rows:

    print(
        f"  Row {index + 1}: "
        f"{row.get('pair_model_1')} vs "
        f"{row.get('pair_model_2')} "
        f"sample {row.get('sample_id')}"
    )


if len(target_rows) != 3:

    raise RuntimeError(
        "Could not find all 3 target rows."
    )


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("\nLoading Phi-3.5-mini...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)


# ============================================================
# 4-BIT CONFIG
# ============================================================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)


# ============================================================
# LOAD MODEL
# ============================================================

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# IMPORTANT
model.config.use_cache = False

model.eval()

print("Model loaded.")


# ============================================================
# BUILD PROMPT
# ============================================================

def build_prompt(row):

    prompt_text = str(
        row.get("prompt", "")
    )

    response_a = str(
        row.get("response_A", "")
    )

    response_b = str(
        row.get("response_B", "")
    )

    model_a = str(
        row.get(
            "displayed_A_model",
            "Response A"
        )
    )

    model_b = str(
        row.get(
            "displayed_B_model",
            "Response B"
        )
    )

    return f"""
You are a strict pairwise evaluator.

Evaluate Response A and Response B for the SAME prompt.

PROMPT:
{prompt_text}

RESPONSE A ({model_a}):
{response_a}

RESPONSE B ({model_b}):
{response_b}

Evaluate these six criteria:

1. Fluency
2. Code-mixing naturalness
3. Hindi grammar
4. Prompt adherence
5. Spelling consistency
6. Overall quality

A = Response A is better
B = Response B is better
T = Tie

Return ONLY exactly six choices.

Example:
A B A T B A

No explanation.
No JSON.
No sentences.
No headings.

Answer:
""".strip()


# ============================================================
# GENERATE
# ============================================================

def generate_judgment(row):

    prompt = build_prompt(row)

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        formatted,
        return_tensors="pt",
        truncation=True,
        max_length=4096
    )

    inputs = {
        k: v.to(model.device)
        for k, v in inputs.items()
    }

    with torch.no_grad():

        outputs = model.generate(
            **inputs,

            max_new_tokens=20,

            do_sample=False,

            use_cache=False,

            pad_token_id=tokenizer.eos_token_id,

            eos_token_id=tokenizer.eos_token_id
        )

    input_length = inputs[
        "input_ids"
    ].shape[1]

    generated = outputs[
        0,
        input_length:
    ]

    raw = tokenizer.decode(
        generated,
        skip_special_tokens=True
    ).strip()

    parsed = parse_judgment(raw)

    return parsed, raw


# ============================================================
# FIX ONLY 3
# ============================================================

for number, (index, row) in enumerate(
    target_rows,
    start=1
):

    print()
    print("-" * 70)

    print(
        f"Fixing {number}/3"
    )

    print(
        f"Pair: {row.get('pair_model_1')} "
        f"vs {row.get('pair_model_2')}"
    )

    print(
        f"Sample: {row.get('sample_id')}"
    )

    # Try up to 3 times if model gives invalid output
    success = False

    for attempt in range(1, 4):

        print(
            f"Attempt {attempt}/3..."
        )

        try:

            parsed, raw = generate_judgment(
                row
            )

            print(
                f"Raw output: {raw!r}"
            )

            if parsed is not None:

                row["overall"] = parsed

                row["judge_output"] = parsed

                row["raw_judge_output"] = raw

                row["recovered"] = True

                print(
                    f"VALID: {parsed}"
                )

                success = True

                break

            else:

                print(
                    "Invalid output. Retrying..."
                )

        except Exception as e:

            print(
                f"Error: {type(e).__name__}: {e}"
            )

            torch.cuda.empty_cache()


    if not success:

        raise RuntimeError(
            f"Could not fix row {index + 1}"
        )


# ============================================================
# FINAL VALIDATION
# ============================================================

valid = 0
invalid = 0

for row in rows:

    parsed = parse_judgment(
        row.get("overall")
    )

    if parsed is not None:

        row["overall"] = parsed

        valid += 1

    else:

        invalid += 1


print()
print("=" * 70)
print("FINAL CHECK")
print("=" * 70)

print(
    f"Total judgments : {len(rows)}"
)

print(
    f"Valid judgments : {valid}"
)

print(
    f"Invalid         : {invalid}"
)


# ============================================================
# SAVE FINAL JSONL
# ============================================================

with open(
    FINAL_JSONL,
    "w",
    encoding="utf-8"
) as f:

    for row in rows:

        f.write(
            json.dumps(
                row,
                ensure_ascii=False
            )
            + "\n"
        )


# ============================================================
# SAVE FINAL CSV
# ============================================================

df = pd.DataFrame(rows)

df.to_csv(
    FINAL_CSV,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# SUCCESS
# ============================================================

print()

if (
    len(rows) == 204
    and valid == 204
    and invalid == 0
):

    print("=" * 70)
    print(
        "SUCCESS: ALL 204 JUDGMENTS ARE VALID"
    )
    print("=" * 70)

else:

    print("=" * 70)
    print(
        "STILL INVALID - CHECK OUTPUT"
    )
    print("=" * 70)


print()
print("Final JSONL:")
print(FINAL_JSONL)

print()
print("Final CSV:")
print(FINAL_CSV)

print()
print("DONE.")