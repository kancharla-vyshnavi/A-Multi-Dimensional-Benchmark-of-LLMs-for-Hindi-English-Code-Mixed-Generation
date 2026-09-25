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

INPUT_CSV = Path(
    "outputs/controlled_1000/benchmark_generations_1000.csv"
)

OUTPUT_DIR = Path(
    "outputs/robustness_mistral_1000"
)

OUTPUT_CSV = OUTPUT_DIR / "mistral_robustness_1000_results.csv"
OUTPUT_JSONL = OUTPUT_DIR / "mistral_robustness_1000_results.jsonl"
CONFIG_JSON = OUTPUT_DIR / "robustness_config_1000.json"

JUDGE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"


# ------------------------------------------------------------
# DO NOT CHANGE THESE
# ------------------------------------------------------------

BATCH_SIZE = 2
MAX_INPUT_TOKENS = 1024
MAX_NEW_TOKENS = 160

SEED = 42


SCORE_FIELDS = [
    "fluency",
    "code_mixing_naturalness",
    "hindi_grammar",
    "prompt_adherence",
    "spelling_consistency",
    "overall",
]


# ============================================================
# PERTURBED ROBUSTNESS RUBRIC
# ============================================================

PERTURBED_PROMPT = r"""
You are an independent evaluator for a Hinglish generation benchmark.

Evaluate ONLY the response text against the original prompt.

Do NOT identify, infer, favor, or penalize any model based on its name.
Do NOT compare the response with other models.
Evaluate every dimension independently.

Use a 1-5 integer scale:

1 = very poor
2 = poor
3 = acceptable
4 = good
5 = excellent

Dimensions:

fluency:
How natural, readable, and fluent is the response?

code_mixing_naturalness:
How naturally does the response combine Hindi and English?
Natural Hinglish does not require excessive English or excessive Hindi.
Do not penalize a response simply because it contains more Hindi or more English.
Focus on whether the switching is natural in context.

hindi_grammar:
How grammatically correct is the Hindi used in the response?
Judge Hindi grammar only where Hindi is actually used.

prompt_adherence:
How well does the response follow the original prompt and preserve its intended meaning?

spelling_consistency:
How consistent and appropriate is the written spelling/transliteration in the response?

overall:
Your overall assessment of the response quality.

IMPORTANT EVALUATION RULES:

- Judge the actual response text.
- Do not make assumptions about the underlying model.
- Do not compare models.
- Do not let one dimension automatically determine another.
- A fluent response can still have grammar or spelling problems.
- A response can have good spelling but poor prompt adherence.
- Judge Hindi grammar only where Hindi occurs.
- Natural code-mixing does not mean maximum language switching.
- Distinguish natural code-mixing from unnecessary or awkward switching.
- Do not reward or penalize short responses merely for length.
- Do not reward or penalize long responses merely for length.
- Evaluate the response as written.
- Do not rewrite or correct the response.
- Do not provide explanations.

Return ONLY valid JSON.

Required format:

{
  "fluency": 1,
  "code_mixing_naturalness": 1,
  "hindi_grammar": 1,
  "prompt_adherence": 1,
  "spelling_consistency": 1,
  "overall": 1
}
"""


# ============================================================
# PERFORMANCE SETTINGS
# ============================================================

def setup_performance():

    # Reproducibility
    torch.manual_seed(SEED)

    if torch.cuda.is_available():

        # Safe CUDA performance settings
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        print("=" * 60)
        print("GPU INFORMATION")
        print("=" * 60)

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

        print(
            "CUDA:",
            torch.version.cuda
        )

        print(
            "VRAM:",
            round(
                torch.cuda.get_device_properties(0).total_memory
                / (1024 ** 3),
                2
            ),
            "GB"
        )

        print("=" * 60)

    else:

        print("WARNING: CUDA is NOT available.")
        print("This run will be extremely slow.")


# ============================================================
# JSON CLEANING
# ============================================================

def clean_json_text(text):

    if not text:
        return ""

    text = text.strip()

    text = re.sub(
        r"```(?:json)?",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = text.replace("```", "").strip()

    return text


# ============================================================
# PARSE SCORES
# ============================================================

def parse_scores(text):

    text = clean_json_text(text)

    # --------------------------------------------------------
    # METHOD 1: DIRECT JSON
    # --------------------------------------------------------

    try:

        obj = json.loads(text)

        if isinstance(obj, dict):

            result = {}

            for field in SCORE_FIELDS:

                value = obj.get(field)

                if isinstance(value, bool):
                    return None

                value = int(value)

                if value < 1 or value > 5:
                    return None

                result[field] = value

            return result

    except Exception:
        pass


    # --------------------------------------------------------
    # METHOD 2: JSON OBJECT INSIDE OUTPUT
    # --------------------------------------------------------

    candidates = re.findall(
        r"\{.*?\}",
        text,
        flags=re.DOTALL
    )

    for candidate in candidates:

        try:

            obj = json.loads(candidate)

            if not isinstance(obj, dict):
                continue

            result = {}

            for field in SCORE_FIELDS:

                value = obj.get(field)

                if isinstance(value, bool):
                    raise ValueError

                value = int(value)

                if value < 1 or value > 5:
                    raise ValueError

                result[field] = value

            return result

        except Exception:
            continue


    # --------------------------------------------------------
    # METHOD 3: REGEX RECOVERY
    # --------------------------------------------------------

    result = {}

    for field in SCORE_FIELDS:

        pattern = (
            rf'"?{re.escape(field)}"?'
            rf'\s*[:=]\s*([1-5])'
        )

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        if not match:
            return None

        result[field] = int(match.group(1))

    return result


# ============================================================
# BUILD PROMPT
# ============================================================

def build_prompt(prompt, response):

    return f"""
{PERTURBED_PROMPT}

ORIGINAL PROMPT:
{prompt}

RESPONSE TO EVALUATE:
{response}

Return ONLY the JSON object.
""".strip()


# ============================================================
# SAVE OUTPUTS
# ============================================================

def save_outputs(rows):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    fieldnames = [
        "row_index",
        "model",
        "prompt",
        "response",
        *SCORE_FIELDS,
        "judge_model",
        "raw_judge_output",
    ]

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    temp_csv = OUTPUT_DIR / "robustness_temp.csv"

    with open(
        temp_csv,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)

    temp_csv.replace(OUTPUT_CSV)


    # --------------------------------------------------------
    # JSONL
    # --------------------------------------------------------

    with open(
        OUTPUT_JSONL,
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
# LOAD EXISTING RESULTS
# ============================================================

def load_existing():

    if not OUTPUT_CSV.exists():
        return {}

    existing = {}

    try:

        with open(
            OUTPUT_CSV,
            newline="",
            encoding="utf-8"
        ) as f:

            reader = csv.DictReader(f)

            if not reader.fieldnames:

                print(
                    "Existing robustness file has no header."
                )

                return {}

            if (
                "row_index" not in reader.fieldnames
                or "model" not in reader.fieldnames
            ):

                print(
                    "Existing robustness file has "
                    "an incompatible header."
                )

                print(
                    "Starting robustness run from scratch."
                )

                return {}

            for row in reader:

                if not row.get("row_index"):
                    continue

                if not row.get("model"):
                    continue

                key = (
                    str(row["model"]),
                    str(row["row_index"])
                )

                existing[key] = row

    except Exception as e:

        print(
            "Could not load existing robustness file:",
            repr(e)
        )

        return {}

    print(
        f"Resume file found: {len(existing)} rows"
    )

    return existing


# ============================================================
# LOAD MODEL
# ============================================================

def load_judge():

    print()
    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        JUDGE_MODEL,
        use_fast=True
    )

    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    print("Loading Mistral judge...")


    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )


    model = AutoModelForCausalLM.from_pretrained(
        JUDGE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        attn_implementation="eager",
    )


    model.eval()

    print("Judge loaded.")
    print()

    return tokenizer, model


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # PERFORMANCE SETUP
    # --------------------------------------------------------

    setup_performance()


    # --------------------------------------------------------
    # OUTPUT DIRECTORY
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------------------
    # LOAD 4000 GENERATIONS
    # --------------------------------------------------------

    print()
    print("Loading benchmark generations...")

    with open(
        INPUT_CSV,
        newline="",
        encoding="utf-8"
    ) as f:

        reader = csv.DictReader(f)

        source_rows = list(reader)


    print(
        f"Loaded generations: {len(source_rows)}"
    )


    if len(source_rows) != 4000:

        print(
            "WARNING:",
            f"Expected 4000 rows, found {len(source_rows)}"
        )


    # --------------------------------------------------------
    # LOAD EXISTING RESULTS
    # --------------------------------------------------------

    existing = load_existing()

    rows = dict(existing)


    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    tokenizer, model = load_judge()


    # --------------------------------------------------------
    # PROCESS
    # --------------------------------------------------------

    total = len(source_rows)

    completed_before = len(rows)

    print()
    print("=" * 60)
    print("ROBUSTNESS EVALUATION STARTED")
    print("=" * 60)
    print(
        f"Total source rows : {total}"
    )
    print(
        f"Already completed : {completed_before}"
    )
    print(
        f"Remaining         : {total - completed_before}"
    )
    print(
        f"Batch size        : {BATCH_SIZE}"
    )
    print(
        f"Max input tokens  : {MAX_INPUT_TOKENS}"
    )
    print(
        f"Max new tokens    : {MAX_NEW_TOKENS}"
    )
    print("=" * 60)
    print()


    start_time = time.time()

    completed_count = len(rows)


    # --------------------------------------------------------
    # BATCH LOOP
    # --------------------------------------------------------

    for start in range(
        0,
        total,
        BATCH_SIZE
    ):

        batch = source_rows[
            start:start + BATCH_SIZE
        ]


        pending = []


        for item in batch:

            key = (
                str(item["model"]),
                str(item["row_index"])
            )

            if key in rows:
                continue

            pending.append(item)


        # ----------------------------------------------------
        # ALREADY PROCESSED
        # ----------------------------------------------------

        if not pending:
            continue


        print(
            f"[{min(start + BATCH_SIZE, total)}/{total}] "
            f"Judging {len(pending)}"
        )


        # ----------------------------------------------------
        # BUILD PROMPTS
        # ----------------------------------------------------

        prompts = []

        for item in pending:

            prompts.append(
                build_prompt(
                    item["prompt"],
                    item["response"]
                )
            )


        # ----------------------------------------------------
        # TOKENIZE
        # ----------------------------------------------------

        try:

            encoded = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_INPUT_TOKENS,
            )


            # Move tensors to model device
            encoded = {
                key: value.to(
                    model.device,
                    non_blocking=True
                )
                for key, value in encoded.items()
            }


            padded_input_length = (
                encoded["input_ids"].shape[1]
            )


            # ------------------------------------------------
            # GENERATE
            # ------------------------------------------------

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


            # ------------------------------------------------
            # REMOVE PROMPT TOKENS
            # ------------------------------------------------

            generated_tokens = generated[
                :,
                padded_input_length:
            ]


            # ------------------------------------------------
            # DECODE
            # ------------------------------------------------

            decoded = tokenizer.batch_decode(
                generated_tokens,
                skip_special_tokens=True
            )


        except torch.cuda.OutOfMemoryError as e:

            print()
            print("CUDA OUT OF MEMORY")
            print(
                "Batch skipped. Existing results are preserved."
            )

            print(repr(e))

            torch.cuda.empty_cache()

            time.sleep(2)

            continue


        except Exception as e:

            print()
            print(
                "BATCH ERROR:",
                repr(e)
            )

            print(
                "Skipping batch."
            )

            torch.cuda.empty_cache()

            time.sleep(1)

            continue


        # ----------------------------------------------------
        # PARSE JUDGMENTS
        # ----------------------------------------------------

        for item, raw_output in zip(
            pending,
            decoded
        ):

            scores = parse_scores(
                raw_output
            )


            row = {
                "row_index": item["row_index"],
                "model": item["model"],
                "prompt": item["prompt"],
                "response": item["response"],
                "judge_model": JUDGE_MODEL,
                "raw_judge_output": raw_output,
            }


            # ------------------------------------------------
            # VALID
            # ------------------------------------------------

            if scores is not None:

                for field in SCORE_FIELDS:
                    row[field] = scores[field]

                print(
                    f"  OK: "
                    f"{item['model']} / "
                    f"{item['row_index']}"
                )


                completed_count += 1


            # ------------------------------------------------
            # INVALID
            # ------------------------------------------------

            else:

                for field in SCORE_FIELDS:
                    row[field] = ""

                print(
                    f"  INVALID: "
                    f"{item['model']} / "
                    f"{item['row_index']}"
                )


            key = (
                str(item["model"]),
                str(item["row_index"])
            )


            rows[key] = row


        # ----------------------------------------------------
        # SAVE AFTER EVERY BATCH
        # ----------------------------------------------------

        save_outputs(
            list(rows.values())
        )


        # ----------------------------------------------------
        # CUDA CACHE
        # ----------------------------------------------------

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        elapsed = time.time() - start_time

        if completed_count > completed_before:

            processed_now = (
                completed_count - completed_before
            )

            rate = (
                processed_now / elapsed
                if elapsed > 0
                else 0
            )

            remaining = total - len(rows)

            eta_seconds = (
                remaining / rate
                if rate > 0
                else 0
            )

            print(
                f"Progress: {len(rows)}/{total} | "
                f"Speed: {rate:.2f} rows/sec | "
                f"ETA: {eta_seconds / 60:.1f} min"
            )


    # ========================================================
    # FINAL CONFIG
    # ========================================================

    config = {

        "input_csv": str(INPUT_CSV),

        "judge_model": JUDGE_MODEL,

        "batch_size": BATCH_SIZE,

        "max_input_tokens": MAX_INPUT_TOKENS,

        "max_new_tokens": MAX_NEW_TOKENS,

        "seed": SEED,

        "score_fields": SCORE_FIELDS,

        "total_rows": len(rows),

        "description": (
            "Robustness evaluation using a "
            "perturbed Mistral evaluation rubric "
            "on the same 4000 benchmark generations."
        ),
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


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    valid_count = 0

    for row in rows.values():

        valid = True

        for field in SCORE_FIELDS:

            if not row.get(field):

                valid = False
                break


        if valid:
            valid_count += 1


    print()
    print("=" * 60)
    print("ROBUSTNESS JUDGING COMPLETE")
    print("=" * 60)

    print(
        f"Total rows: {len(rows)}"
    )

    print(
        f"Valid judgments: {valid_count}"
    )

    print(
        f"Invalid/missing: "
        f"{len(rows) - valid_count}"
    )

    print()

    print(
        f"CSV: {OUTPUT_CSV}"
    )

    print(
        f"JSONL: {OUTPUT_JSONL}"
    )

    print(
        f"Config: {CONFIG_JSON}"
    )

    print("=" * 60)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()