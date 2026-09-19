import os
import json
import re
import random
import warnings

import numpy as np
import pandas as pd
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

from transformers.cache_utils import DynamicCache


# ============================================================
# CONFIG
# ============================================================

JUDGE_MODEL = "microsoft/Phi-3.5-mini-instruct"

INPUT_FILE = r".\outputs\error_analysis\manual_error_annotation_final.csv"

OUTPUT_FILE = r".\outputs\error_analysis\manual_error_annotation_final.csv"

CHECKPOINT_FILE = r".\outputs\error_analysis\semantic_error_checkpoint.csv"

MAX_INPUT_LENGTH = 768
MAX_NEW_TOKENS = 80

SEED = 42


# ============================================================
# WARNINGS
# ============================================================

warnings.filterwarnings(
    "ignore",
    category=FutureWarning
)


# ============================================================
# PHI-3 / TRANSFORMERS COMPATIBILITY
# ============================================================

# Older Phi-3 remote code expects older DynamicCache APIs.
# New Transformers uses different method names/signatures.

if not hasattr(DynamicCache, "seen_tokens"):

    DynamicCache.seen_tokens = property(
        lambda self: self.get_seq_length()
    )


if not hasattr(DynamicCache, "get_usable_length"):

    def get_usable_length_compat(
        self,
        *args,
        **kwargs
    ):
        return self.get_seq_length()

    DynamicCache.get_usable_length = (
        get_usable_length_compat
    )


# ============================================================
# SEED
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# ERROR DEFINITIONS
# ============================================================

ERROR_DEFINITIONS = """
E1 = English-dominant output
E2 = Hindi-dominant output
E3 = Unnatural code-switching
E4 = Grammatical error
E5 = Repetition
E6 = Prompt misunderstanding
E7 = Incomplete response
E8 = Spelling/transliteration error
E9 = Hallucination/factual error
E10 = Irrelevant response
"""


# ============================================================
# ERROR COLUMNS
# ============================================================

ERROR_COLUMNS = {
    "E1": "E1_English_dominant",
    "E2": "E2_Hindi_dominant",
    "E3": "E3_Unnatural_code_switching",
    "E4": "E4_Grammatical_error",
    "E5": "E5_Repetition",
    "E6": "E6_Prompt_misunderstanding",
    "E7": "E7_Incomplete_response",
    "E8": "E8_Spelling_transliteration_error",
    "E9": "E9_Hallucination_factual_error",
    "E10": "E10_Irrelevant_response",
}


# ============================================================
# START
# ============================================================

print("=" * 70)
print("SEMANTIC ERROR ANALYSIS")
print("=" * 70)


# ============================================================
# GPU
# ============================================================

if not torch.cuda.is_available():

    raise RuntimeError(
        "CUDA GPU is required."
    )

print(
    "GPU:",
    torch.cuda.get_device_name(0)
)


# ============================================================
# LOAD CSV
# ============================================================

if not os.path.exists(INPUT_FILE):

    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(
    INPUT_FILE
)

print(
    "\nInput shape:",
    df.shape
)

if len(df) != 136:

    raise ValueError(
        f"Expected 136 rows, found {len(df)}"
    )


# ============================================================
# RESTORE CHECKPOINT
# ============================================================

if os.path.exists(CHECKPOINT_FILE):

    try:

        checkpoint_df = pd.read_csv(
            CHECKPOINT_FILE
        )

        if len(checkpoint_df) == 136:

            df = checkpoint_df

            print(
                "Checkpoint restored successfully."
            )

    except Exception as e:

        print(
            "Checkpoint restore failed:",
            e
        )


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "model",
    "sample",
    "category",
    "prompt",
    "response"
]

for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"Missing column: {column}"
        )


# ============================================================
# CREATE ERROR COLUMNS
# ============================================================

for column in ERROR_COLUMNS.values():

    if column not in df.columns:

        df[column] = 0


if "Final_Error_Category" not in df.columns:

    df["Final_Error_Category"] = ""


if "Manual_Notes" not in df.columns:

    df["Manual_Notes"] = ""


if "Review_Status" not in df.columns:

    df["Review_Status"] = "NOT_REVIEWED"


for column in [
    "Final_Error_Category",
    "Manual_Notes",
    "Review_Status"
]:

    df[column] = (
        df[column]
        .fillna("")
        .astype(str)
    )


# ============================================================
# STATUS
# ============================================================

manual_count = (
    df["Review_Status"]
    .str.upper()
    .eq("REVIEWED")
    .sum()
)

semantic_count = (
    df["Review_Status"]
    .str.upper()
    .eq("SEMANTIC_BATCH_REVIEWED")
    .sum()
)

print(
    "\nManually reviewed:",
    manual_count
)

print(
    "Already semantic reviewed:",
    semantic_count
)

print(
    "Remaining:",
    136 - manual_count - semantic_count
)


# ============================================================
# LOAD MODEL
# ============================================================

print(
    "\nLoading Phi-3.5-mini-Instruct..."
)

bnb_config = BitsAndBytesConfig(

    load_in_4bit=True,

    bnb_4bit_quant_type="nf4",

    bnb_4bit_compute_dtype=torch.float16,

    bnb_4bit_use_double_quant=True
)


tokenizer = AutoTokenizer.from_pretrained(

    JUDGE_MODEL,

    trust_remote_code=True
)


if tokenizer.pad_token is None:

    tokenizer.pad_token = tokenizer.eos_token


model = AutoModelForCausalLM.from_pretrained(

    JUDGE_MODEL,

    quantization_config=bnb_config,

    device_map="auto",

    trust_remote_code=True
)


model.eval()


print(
    "Judge model loaded successfully."
)


# ============================================================
# BUILD PROMPT
# ============================================================

def build_prompt(
    model_name,
    sample,
    prompt,
    response
):

    return f"""
You are an expert evaluator for a Hindi-English code-mixed
(Hinglish) LLM evaluation study.

MODEL: {model_name}

SAMPLE: {sample}

ORIGINAL PROMPT:
{prompt}

MODEL RESPONSE:
{response}

ERROR CATEGORIES:

{ERROR_DEFINITIONS}

Rules:

1. Compare the response with the original prompt.
2. Only assign clearly supported errors.
3. Multiple errors are allowed.
4. If there is no clear error, return an empty list.
5. E1/E2 require clear language dominance.
6. E3 means unnatural Hindi-English mixing.
7. E4 means clear grammatical errors.
8. E5 means unnecessary repetition.
9. E6 means failure to understand/follow the prompt.
10. E7 means genuinely incomplete response.
11. E8 means clear spelling/transliteration errors.
12. E9 means clearly identifiable factual error.
13. E10 means irrelevant content.
14. Be conservative.

Return ONLY JSON.

Format:

{{"errors":["E3","E6"],"notes":"Brief evidence."}}

No errors:

{{"errors":[],"notes":"No clear error."}}
"""


# ============================================================
# JSON EXTRACTION
# ============================================================

def extract_json(text):

    if not isinstance(text, str):

        return None

    text = text.strip()


    # Direct JSON
    try:

        return json.loads(text)

    except Exception:

        pass


    # Markdown JSON
    match = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```",
        text,
        flags=re.DOTALL | re.IGNORECASE
    )

    if match:

        try:

            return json.loads(
                match.group(1)
            )

        except Exception:

            pass


    # Find JSON object
    start = text.find("{")

    end = text.rfind("}")

    if start != -1 and end > start:

        try:

            return json.loads(
                text[start:end + 1]
            )

        except Exception:

            pass


    return None


# ============================================================
# ERROR CODE EXTRACTION
# ============================================================

def extract_error_codes(text):

    if not isinstance(text, str):

        return []

    codes = re.findall(
        r"\bE(?:10|[1-9])\b",
        text.upper()
    )

    codes = [
        code
        for code in codes
        if code in ERROR_COLUMNS
    ]

    return list(
        dict.fromkeys(codes)
    )


# ============================================================
# PARSE RESULT
# ============================================================

def parse_result(raw_output):

    result = extract_json(
        raw_output
    )


    if isinstance(result, dict):

        errors = result.get(
            "errors",
            []
        )

        if not isinstance(errors, list):

            errors = []


        errors = [

            str(code)
            .strip()
            .upper()

            for code in errors

        ]


        errors = [

            code

            for code in errors

            if code in ERROR_COLUMNS

        ]


        errors = list(
            dict.fromkeys(errors)
        )


        notes = result.get(
            "notes",
            ""
        )


        if notes is None:

            notes = ""


        return (
            errors,
            str(notes).strip()
        )


    # Fallback
    errors = extract_error_codes(
        raw_output
    )


    if errors:

        return (
            errors,
            "Error codes extracted from semantic judge output."
        )


    return (
        [],
        "No parseable error codes; manual verification recommended."
    )


# ============================================================
# RUN JUDGE
# ============================================================

def run_judge(judge_prompt):

    messages = [

        {
            "role": "user",
            "content": judge_prompt
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

        max_length=MAX_INPUT_LENGTH
    )


    inputs = {

        key: value.to(model.device)

        for key, value in inputs.items()

    }


    with torch.inference_mode():

        output = model.generate(

            **inputs,

            max_new_tokens=MAX_NEW_TOKENS,

            do_sample=False,

            use_cache=True,

            pad_token_id=tokenizer.pad_token_id,

            eos_token_id=tokenizer.eos_token_id
        )


    generated_tokens = output[

        0,

        inputs["input_ids"].shape[1]:

    ]


    return tokenizer.decode(

        generated_tokens,

        skip_special_tokens=True
    )


# ============================================================
# SAVE CHECKPOINT
# ============================================================

def save_checkpoint():

    df.to_csv(

        CHECKPOINT_FILE,

        index=False,

        encoding="utf-8-sig"
    )


# ============================================================
# START REVIEW
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "STARTING SEMANTIC REVIEW"
)

print(
    "=" * 70
)


# ============================================================
# PROCESS ALL 136
# ============================================================

for idx in range(136):

    status = str(

        df.loc[
            idx,
            "Review_Status"
        ]

    ).strip().upper()


    # --------------------------------------------------------
    # PRESERVE MANUAL REVIEWS
    # --------------------------------------------------------

    if status == "REVIEWED":

        print(

            f"[{idx + 1}/136] "

            f"{df.loc[idx, 'model']} "

            f"sample {df.loc[idx, 'sample']} "

            f"-> PRESERVED REVIEWED"

        )

        continue


    # --------------------------------------------------------
    # SKIP ALREADY PROCESSED
    # --------------------------------------------------------

    if status == "SEMANTIC_BATCH_REVIEWED":

        print(

            f"[{idx + 1}/136] "

            f"{df.loc[idx, 'model']} "

            f"sample {df.loc[idx, 'sample']} "

            f"-> ALREADY PROCESSED"

        )

        continue


    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    model_name = str(
        df.loc[idx, "model"]
    )

    sample = str(
        df.loc[idx, "sample"]
    )

    prompt = str(
        df.loc[idx, "prompt"]
    )

    response = str(
        df.loc[idx, "response"]
    )


    print(

        f"\n[{idx + 1}/136] "

        f"{model_name} | Sample {sample}"

    )


    # --------------------------------------------------------
    # BUILD PROMPT
    # --------------------------------------------------------

    judge_prompt = build_prompt(

        model_name,

        sample,

        prompt,

        response
    )


    # --------------------------------------------------------
    # GENERATE
    # --------------------------------------------------------

    try:

        raw_output = run_judge(
            judge_prompt
        )


    except KeyboardInterrupt:

        print(
            "\nInterrupted."
        )

        print(
            "Saving checkpoint..."
        )

        save_checkpoint()

        raise


    except Exception as e:

        print(
            "\nGeneration error:"
        )

        print(
            repr(e)
        )

        print(
            "\nCheckpoint saved."
        )

        save_checkpoint()

        raise


    # --------------------------------------------------------
    # PARSE
    # --------------------------------------------------------

    errors, notes = parse_result(
        raw_output
    )


    # --------------------------------------------------------
    # WRITE E1-E10
    # --------------------------------------------------------

    for code, column in ERROR_COLUMNS.items():

        df.loc[
            idx,
            column
        ] = (

            1
            if code in errors
            else 0

        )


    # --------------------------------------------------------
    # FINAL CATEGORY
    # --------------------------------------------------------

    if errors:

        df.loc[
            idx,
            "Final_Error_Category"
        ] = ",".join(errors)

    else:

        df.loc[
            idx,
            "Final_Error_Category"
        ] = "NONE"


    # --------------------------------------------------------
    # NOTES
    # --------------------------------------------------------

    df.loc[
        idx,
        "Manual_Notes"
    ] = notes


    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    df.loc[
        idx,
        "Review_Status"
    ] = "SEMANTIC_BATCH_REVIEWED"


    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print(

        "  Errors:",

        ",".join(errors)

        if errors

        else "NONE"

    )


    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    save_checkpoint()


# ============================================================
# FINAL SAVE
# ============================================================

df.to_csv(

    OUTPUT_FILE,

    index=False,

    encoding="utf-8-sig"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "SEMANTIC ERROR ANALYSIS COMPLETED"
)

print(
    "=" * 70
)

print(
    "\nTotal responses:",
    len(df)
)


print(
    "\nModel counts:"
)

print(
    df["model"].value_counts()
)


print(
    "\nReview status:"
)

print(
    df["Review_Status"].value_counts()
)


print(
    "\nError counts:"
)

for code, column in ERROR_COLUMNS.items():

    count = pd.to_numeric(

        df[column],

        errors="coerce"

    ).fillna(0).sum()


    print(
        f"{code}: {int(count)}"
    )


print(
    "\nOutput:"
)

print(
    OUTPUT_FILE
)


print(
    "\nSTATUS: ALL 136 RESPONSES PROCESSED"
)

print(
    "=" * 70
)