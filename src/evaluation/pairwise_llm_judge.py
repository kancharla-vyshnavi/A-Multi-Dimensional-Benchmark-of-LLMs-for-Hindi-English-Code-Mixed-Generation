import os
import re
import json
import pandas as pd
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"

INPUT_JSONL = r".\outputs\pairwise_judge\pairwise_judge_results.jsonl"

RECOVERY_JSONL = r".\outputs\pairwise_judge\pairwise_judge_recovery.jsonl"

FINAL_JSONL = r".\outputs\pairwise_judge\pairwise_judge_results_final.jsonl"

FINAL_CSV = r".\outputs\pairwise_judge\pairwise_judge_results_final.csv"

FINAL_SUMMARY = r".\outputs\pairwise_judge\pairwise_judge_summary_final.csv"


MAX_NEW_TOKENS = 20


# ============================================================
# UNIQUE KEY
# ============================================================

def get_key(row):

    return (
        str(row.get("pair_model_1", "")).strip(),
        str(row.get("pair_model_2", "")).strip(),
        str(row.get("sample_id", "")).strip()
    )


# ============================================================
# PARSE JUDGMENT
# ============================================================

def parse_judgment(value):

    if value is None:
        return None

    if pd.isna(value):
        return None

    text = str(value).strip().upper()

    if not text:
        return None

    # --------------------------------------------------------
    # Remove common formatting
    # --------------------------------------------------------

    text = text.replace("[", " ")
    text = text.replace("]", " ")
    text = text.replace('"', " ")
    text = text.replace("'", " ")

    # --------------------------------------------------------
    # CASE 1:
    # A B B T A A
    # --------------------------------------------------------

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

        if all(x in {"A", "B", "T"} for x in result):

            return " ".join(result)

    # --------------------------------------------------------
    # CASE 2:
    # ABBTAA
    # --------------------------------------------------------

    compact_matches = re.findall(
        r"\b[ABT]{6}\b",
        text
    )

    if compact_matches:

        value = compact_matches[-1]

        return " ".join(list(value))

    # --------------------------------------------------------
    # CASE 3:
    # Search anywhere for 6 consecutive A/B/T
    # --------------------------------------------------------

    compact = re.sub(
        r"[^ABT]",
        "",
        text
    )

    if len(compact) >= 6:

        candidate = compact[-6:]

        if all(x in {"A", "B", "T"} for x in candidate):

            return " ".join(list(candidate))

    # --------------------------------------------------------
    # CASE 4:
    # Existing old JSON-style results
    #
    # Example:
    # "overall": "A"
    #
    # Not enough by itself, so do NOT mark valid.
    # --------------------------------------------------------

    return None


# ============================================================
# VALIDATION
# ============================================================

def is_valid_judgment(row):

    # First check explicit overall field
    overall = row.get("overall")

    parsed = parse_judgment(overall)

    if parsed is not None:
        return True

    # --------------------------------------------------------
    # Some previous rows may store the actual six choices in
    # judge_output / raw_judge_output.
    # --------------------------------------------------------

    for field in [
        "judge_output",
        "raw_judge_output",
        "response"
    ]:

        if field in row:

            parsed = parse_judgment(
                row.get(field)
            )

            if parsed is not None:
                return True

    return False


# ============================================================
# LOAD ORIGINAL JSONL
# ============================================================

print("=" * 70)
print("PAIRWISE JUDGE - SAFE RECOVERY")
print("=" * 70)

if not os.path.exists(INPUT_JSONL):

    raise FileNotFoundError(
        f"\nOriginal JSONL not found:\n{INPUT_JSONL}"
    )


original_rows = []

with open(
    INPUT_JSONL,
    "r",
    encoding="utf-8"
) as f:

    for line in f:

        line = line.strip()

        if not line:
            continue

        try:

            row = json.loads(line)

            original_rows.append(row)

        except Exception:
            pass


print()
print(f"Original rows loaded: {len(original_rows)}")


# ============================================================
# IDENTIFY EXISTING VALID / INVALID
# ============================================================

valid_original = {}
invalid_original = {}


for row in original_rows:

    key = get_key(row)

    if is_valid_judgment(row):

        valid_original[key] = row

    else:

        invalid_original[key] = row


print(
    f"Existing valid   : {len(valid_original)}"
)

print(
    f"Existing invalid : {len(invalid_original)}"
)


# ============================================================
# LOAD PREVIOUS RECOVERY RESULTS
# ============================================================

recovered = {}


if os.path.exists(RECOVERY_JSONL):

    with open(
        RECOVERY_JSONL,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            try:

                row = json.loads(line)

                parsed = parse_judgment(
                    row.get("overall")
                )

                if parsed is not None:

                    row["overall"] = parsed

                    recovered[
                        get_key(row)
                    ] = row

            except Exception:
                pass


print(
    f"Already recovered: {len(recovered)}"
)


# ============================================================
# CREATE JOB LIST
# ============================================================

jobs = []


for key, row in invalid_original.items():

    if key not in recovered:

        jobs.append(row)


print(
    f"Remaining to rerun: {len(jobs)}"
)


# ============================================================
# LOAD MODEL ONLY IF NEEDED
# ============================================================

model = None
tokenizer = None


if len(jobs) > 0:

    print()
    print("Loading Phi-3.5-mini judge...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )

    # --------------------------------------------------------
    # PROPER 4-BIT CONFIGURATION
    # --------------------------------------------------------

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    # IMPORTANT:
    # Prevent DynamicCache / seen_tokens issue
    model.config.use_cache = False

    model.eval()

    print("Model loaded successfully.")


# ============================================================
# CREATE RECOVERY FILE
# ============================================================

os.makedirs(
    os.path.dirname(RECOVERY_JSONL),
    exist_ok=True
)

if not os.path.exists(RECOVERY_JSONL):

    with open(
        RECOVERY_JSONL,
        "w",
        encoding="utf-8"
    ):
        pass


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

    prompt = f"""
You are a strict pairwise evaluator.

Evaluate Response A and Response B for the SAME prompt.

PROMPT:
{prompt_text}

RESPONSE A ({model_a}):
{response_a}

RESPONSE B ({model_b}):
{response_b}

Evaluate these 6 criteria:

1. Fluency
2. Code-mixing naturalness
3. Hindi grammar
4. Prompt adherence
5. Spelling consistency
6. Overall quality

For each criterion:

A = Response A is better
B = Response B is better
T = Tie

YOUR OUTPUT MUST CONTAIN EXACTLY SIX LETTERS.

Example:
A B A T B A

IMPORTANT:
- Output ONLY six choices.
- Use only A, B, or T.
- Separate choices with spaces.
- No explanation.
- No JSON.
- No sentences.
- No headings.
- No analysis.

Answer:
""".strip()

    return prompt


# ============================================================
# RUN ONE JUDGMENT
# ============================================================

def run_judgment(row):

    prompt = build_prompt(row)

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
        max_length=4096
    )

    # Move tensors to model device
    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model.generate(
            **inputs,

            max_new_tokens=MAX_NEW_TOKENS,

            do_sample=False,

            use_cache=False,

            pad_token_id=tokenizer.eos_token_id,

            eos_token_id=tokenizer.eos_token_id
        )

    input_length = inputs[
        "input_ids"
    ].shape[1]

    generated_tokens = outputs[
        0,
        input_length:
    ]

    raw_output = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    normalized = parse_judgment(
        raw_output
    )

    return normalized, raw_output


# ============================================================
# RECOVERY LOOP
# ============================================================

if len(jobs) > 0:

    print()
    print("=" * 70)
    print("STARTING RECOVERY")
    print("=" * 70)

    print()
    print(
        f"Jobs to process: {len(jobs)}"
    )

    print(
        "Every valid result will be saved immediately."
    )

    print()

    for index, row in enumerate(
        jobs,
        start=1
    ):

        key = get_key(row)

        try:

            normalized, raw_output = run_judgment(
                row
            )

            # ------------------------------------------------
            # SUCCESS
            # ------------------------------------------------

            if normalized is not None:

                new_row = dict(row)

                new_row["overall"] = normalized

                new_row["judge_output"] = normalized

                new_row["raw_judge_output"] = raw_output

                new_row["recovered"] = True

                # --------------------------------------------
                # SAVE IMMEDIATELY
                # --------------------------------------------

                with open(
                    RECOVERY_JSONL,
                    "a",
                    encoding="utf-8"
                ) as f:

                    f.write(
                        json.dumps(
                            new_row,
                            ensure_ascii=False
                        )
                        + "\n"
                    )

                recovered[key] = new_row

                print(
                    f"[{index}/{len(jobs)}] "
                    f"VALID -> {normalized}"
                )

            else:

                print(
                    f"[{index}/{len(jobs)}] "
                    f"INVALID -> {raw_output[:80]!r}"
                )

        except KeyboardInterrupt:

            print()
            print("=" * 70)
            print("STOPPED BY USER")
            print("=" * 70)

            print()
            print(
                f"Recovered so far: {len(recovered)}"
            )

            print()
            print(
                "Progress saved safely in:"
            )

            print(
                RECOVERY_JSONL
            )

            print()
            print(
                "Run the same command again to continue."
            )

            raise

        except torch.cuda.OutOfMemoryError:

            print()
            print(
                f"[{index}/{len(jobs)}] "
                "CUDA OUT OF MEMORY"
            )

            print(
                "Clearing GPU cache..."
            )

            torch.cuda.empty_cache()

        except Exception as e:

            print(
                f"[{index}/{len(jobs)}] "
                f"ERROR: {type(e).__name__}: {e}"
            )


# ============================================================
# FINAL MERGE
# ============================================================

print()
print("=" * 70)
print("FINAL MERGE")
print("=" * 70)


final_rows = []


for row in original_rows:

    key = get_key(row)

    # Recovered valid result replaces original invalid result
    if key in recovered:

        final_rows.append(
            recovered[key]
        )

    else:

        # Keep original valid result unchanged
        final_rows.append(
            row
        )


# ============================================================
# NORMALIZE FINAL OVERALL VALUES
# ============================================================

for row in final_rows:

    parsed = parse_judgment(
        row.get("overall")
    )

    if parsed is not None:

        row["overall"] = parsed


# ============================================================
# FINAL COUNTS
# ============================================================

final_valid = sum(
    1
    for row in final_rows
    if parse_judgment(
        row.get("overall")
    ) is not None
)

final_invalid = (
    len(final_rows)
    - final_valid
)


print()
print(
    f"Total judgments : {len(final_rows)}"
)

print(
    f"Valid judgments : {final_valid}"
)

print(
    f"Invalid         : {final_invalid}"
)


# ============================================================
# SAVE FINAL JSONL
# ============================================================

with open(
    FINAL_JSONL,
    "w",
    encoding="utf-8"
) as f:

    for row in final_rows:

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

final_df = pd.DataFrame(
    final_rows
)

final_df.to_csv(
    FINAL_CSV,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# PAIRWISE SUMMARY
# ============================================================

summary_rows = []


for row in final_rows:

    parsed = parse_judgment(
        row.get("overall")
    )

    if parsed is None:
        continue

    choices = parsed.split()

    if len(choices) != 6:
        continue

    summary_rows.append(
        {
            "pair_model_1":
                row.get(
                    "pair_model_1",
                    ""
                ),

            "pair_model_2":
                row.get(
                    "pair_model_2",
                    ""
                ),

            "sample_id":
                row.get(
                    "sample_id",
                    ""
                ),

            "fluency":
                choices[0],

            "code_mixing_naturalness":
                choices[1],

            "hindi_grammar":
                choices[2],

            "prompt_adherence":
                choices[3],

            "spelling_consistency":
                choices[4],

            "overall":
                choices[5]
        }
    )


summary_detail_df = pd.DataFrame(
    summary_rows
)


# ============================================================
# SAVE DETAILED SUMMARY
# ============================================================

if len(summary_detail_df) > 0:

    summary_detail_df.to_csv(
        FINAL_SUMMARY,
        index=False,
        encoding="utf-8-sig"
    )

else:

    pd.DataFrame().to_csv(
        FINAL_SUMMARY,
        index=False
    )


# ============================================================
# FINAL STATUS
# ============================================================

print()
print("=" * 70)

if (
    len(final_rows) == 204
    and final_valid == 204
    and final_invalid == 0
):

    print(
        "SUCCESS: ALL 204 JUDGMENTS ARE VALID"
    )

else:

    print(
        "RECOVERY NOT COMPLETE YET"
    )

print("=" * 70)

print()
print("Final JSONL:")
print(FINAL_JSONL)

print()
print("Final CSV:")
print(FINAL_CSV)

print()
print("Final Summary:")
print(FINAL_SUMMARY)

print()
print("Recovery progress:")
print(RECOVERY_JSONL)

print()
print("DONE.")