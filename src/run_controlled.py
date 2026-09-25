import os
import json
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

SEED = 42

BASE_DIR = Path(__file__).resolve().parent

BENCHMARK_PATH = (
    BASE_DIR
    / "data"
    / "benchmarks"
    / "hinglish_bench_1000.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
    / "controlled_1000"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CSV_PATH = (
    OUTPUT_DIR
    / "benchmark_generations_1000.csv"
)

JSONL_PATH = (
    OUTPUT_DIR
    / "benchmark_generations_1000.jsonl"
)

CONFIG_PATH = (
    OUTPUT_DIR
    / "generation_config_1000.json"
)

MAX_INPUT_TOKENS = 512
MAX_NEW_TOKENS = 50

TEMPERATURE = 0.7
TOP_P = 0.9
REPETITION_PENALTY = 1.1

USE_4BIT = True


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=SEED):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed()


# ============================================================
# DEVICE
# ============================================================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 80)
print("CONTROLLED V3 HINGLISH BENCHMARK GENERATION")
print("=" * 80)

print(f"Device       : {DEVICE}")

if torch.cuda.is_available():

    print(
        f"GPU          : "
        f"{torch.cuda.get_device_name(0)}"
    )

    print(
        f"PyTorch      : "
        f"{torch.__version__}"
    )

    print(
        f"CUDA         : "
        f"{torch.version.cuda}"
    )

    try:
        total_memory = (
            torch.cuda.get_device_properties(0)
            .total_memory
            / (1024 ** 3)
        )

        print(
            f"VRAM         : "
            f"{total_memory:.1f} GB"
        )

    except Exception:
        pass

print(f"Seed         : {SEED}")
print(f"Max input    : {MAX_INPUT_TOKENS}")
print(f"Max new      : {MAX_NEW_TOKENS}")
print(f"Temperature  : {TEMPERATURE}")
print(f"Top-p        : {TOP_P}")
print(f"Rep penalty  : {REPETITION_PENALTY}")
print(f"4-bit        : {USE_4BIT}")

print(
    "Qwen2.5-7B mode: DIRECT GPU + GREEDY"
)

print()


# ============================================================
# MODELS
# ============================================================

MODELS = {

    "Phi-3.5-mini": {
        "model_name":
            "microsoft/Phi-3.5-mini-instruct",

        "checkpoint":
            BASE_DIR
            / "models"
            / "phi35_mini_cpt"
            / "checkpoint-17340",

        "type": "chat",
    },

    "Qwen2.5-3B": {
        "model_name":
            "Qwen/Qwen2.5-3B",

        "checkpoint":
            BASE_DIR
            / "models"
            / "qwen25_3b_cpt"
            / "checkpoint-17340",

        "type": "chat",
    },

    "HingGPT": {
        "model_name":
            "l3cube-pune/hing-gpt",

        "checkpoint":
            BASE_DIR
            / "models"
            / "hinggpt_cpt"
            / "checkpoint-17340",

        "type": "hinggpt",
    },

    "Qwen2.5-7B": {
        "model_name":
            "Qwen/Qwen2.5-7B",

        "checkpoint":
            BASE_DIR
            / "models"
            / "qwen25_7b_cpt"
            / "checkpoint-17340",

        "type": "chat",
    },
}


# ============================================================
# GENERATION CONFIG
# ============================================================

generation_config = {

    "seed": SEED,

    "max_input_tokens":
        MAX_INPUT_TOKENS,

    "max_new_tokens":
        MAX_NEW_TOKENS,

    "temperature":
        TEMPERATURE,

    "top_p":
        TOP_P,

    "repetition_penalty":
        REPETITION_PENALTY,

    "use_4bit":
        USE_4BIT,

    "qwen_7b_mode":
        "direct_gpu_greedy",

    "device":
        DEVICE,

    "models": {

        name: {

            "model_name":
                cfg["model_name"],

            "checkpoint":
                str(cfg["checkpoint"]),

            "type":
                cfg["type"],
        }

        for name, cfg in MODELS.items()
    },
}


with open(
    CONFIG_PATH,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        generation_config,
        f,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# LOAD BENCHMARK
# ============================================================

if not BENCHMARK_PATH.exists():

    raise FileNotFoundError(
        "Benchmark file not found:\n"
        f"{BENCHMARK_PATH}"
    )


bench = pd.read_csv(
    BENCHMARK_PATH
)

print("=" * 80)
print("LOADING BENCHMARK")
print("=" * 80)

print(
    f"Benchmark: "
    f"{BENCHMARK_PATH}"
)

print(
    f"Rows: {len(bench)}"
)

print()


# ============================================================
# DETECT PROMPT COLUMN
# ============================================================

possible_prompt_columns = [

    "prompt",
    "Prompt",
    "instruction",
    "text",
    "input",
]

PROMPT_COLUMN = None

for col in possible_prompt_columns:

    if col in bench.columns:

        PROMPT_COLUMN = col
        break


if PROMPT_COLUMN is None:

    raise ValueError(
        "Could not find prompt column.\n"
        f"Available columns: "
        f"{list(bench.columns)}"
    )


print(
    f"Prompt column: "
    f"{PROMPT_COLUMN}"
)

print()


# ============================================================
# LOAD EXISTING V3 RESULTS
# ============================================================

results = []


if CSV_PATH.exists():

    print("=" * 80)
    print("LOADING EXISTING V3 RESULTS")
    print("=" * 80)

    try:

        old_df = pd.read_csv(
            CSV_PATH
        )

        if not old_df.empty:

            results = (
                old_df
                .to_dict("records")
            )

            print(
                f"Existing rows loaded: "
                f"{len(results)}"
            )

            if "model" in old_df.columns:

                print(
                    old_df["model"]
                    .value_counts()
                )

            print()

    except Exception as e:

        print(
            "WARNING: Existing V3 CSV "
            "could not be loaded."
        )

        print(
            repr(e)
        )

        print()


# ============================================================
# NORMALIZE EXISTING RESULTS
# ============================================================

#
# IMPORTANT:
#
# We NEVER require an `id` column.
#
# The benchmark uses row_index as the
# stable identifier.
#
# This fixes:
#
# KeyError: 'id'
#
# ============================================================

def normalize_row_index(value):

    try:

        return int(value)

    except Exception:

        return None


completed = set()


for row in results:

    model_name = str(
        row.get(
            "model",
            ""
        )
    ).strip()

    row_index = normalize_row_index(
        row.get(
            "row_index",
            None
        )
    )

    response = str(
        row.get(
            "response",
            ""
        )
    ).strip()

    if not model_name:
        continue

    if row_index is None:
        continue

    if not response:
        continue

    if response.startswith(
        "[GENERATION_ERROR]"
    ):
        continue

    completed.add(
        (
            model_name,
            row_index,
        )
    )


print(
    f"Completed valid keys: "
    f"{len(completed)}"
)

print()


# ============================================================
# SAVE FUNCTIONS
# ============================================================

def save_results():

    if not results:
        return

    df = pd.DataFrame(
        results
    )

    # Make sure required columns exist.
    for column in [
        "row_index",
        "model",
        "prompt",
        "response",
    ]:

        if column not in df.columns:

            df[column] = ""

    # Convert row index safely.
    df["row_index"] = pd.to_numeric(
        df["row_index"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "row_index"
        ]
    )

    df["row_index"] = (
        df["row_index"]
        .astype(int)
    )

    # Remove duplicate model/row pairs.
    df = df.drop_duplicates(
        subset=[
            "model",
            "row_index",
        ],
        keep="last",
    )

    df = df.sort_values(
        [
            "model",
            "row_index",
        ]
    )

    df = df[
        [
            "row_index",
            "model",
            "prompt",
            "response",
        ]
    ]

    df.to_csv(
        CSV_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    with open(
        JSONL_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        for record in (
            df.to_dict(
                "records"
            )
        ):

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


# ============================================================
# QUANTIZATION
# ============================================================

bnb_config = None


if (
    USE_4BIT
    and torch.cuda.is_available()
):

    bnb_config = BitsAndBytesConfig(

        load_in_4bit=True,

        bnb_4bit_quant_type="nf4",

        bnb_4bit_use_double_quant=True,

        bnb_4bit_compute_dtype=
            torch.float16,
    )


# ============================================================
# MODEL LOADING
# ============================================================

def load_model(model_name, model_cfg):

    checkpoint = Path(
        model_cfg["checkpoint"]
    )

    if not checkpoint.exists():

        raise FileNotFoundError(
            "Checkpoint not found:\n"
            f"{checkpoint}"
        )


    print("-" * 80)

    print(
        f"Loading tokenizer: "
        f"{checkpoint}"
    )


    tokenizer = (
        AutoTokenizer.from_pretrained(
            str(checkpoint),
            trust_remote_code=True,
            use_fast=True,
        )
    )


    if tokenizer.pad_token_id is None:

        if (
            tokenizer.eos_token_id
            is not None
        ):

            tokenizer.pad_token = (
                tokenizer.eos_token
            )

        else:

            tokenizer.add_special_tokens(
                {
                    "pad_token":
                        "[PAD]"
                }
            )


    print(
        f"Loading model: "
        f"{model_name}"
    )


    model_kwargs = {
        "trust_remote_code":
            True,
    }


    if torch.cuda.is_available():

        if USE_4BIT:

            model_kwargs[
                "quantization_config"
            ] = bnb_config


            # ==================================================
            # QWEN 7B:
            #
            # DO NOT use device_map="auto".
            #
            # Earlier V3 failed inside accelerate/bitsandbytes
            # with:
            #
            # Tensor.item() cannot be called on meta tensors
            #
            # Therefore Qwen2.5-7B is explicitly placed
            # on GPU 0.
            # ==================================================

            if model_name == "Qwen2.5-7B":

                model_kwargs[
                    "device_map"
                ] = {
                    "": 0
                }

                model_kwargs[
                    "torch_dtype"
                ] = torch.float16

            else:

                model_kwargs[
                    "device_map"
                ] = "auto"

        else:

            model_kwargs[
                "torch_dtype"
            ] = torch.float16


            if model_name == "Qwen2.5-7B":

                model_kwargs[
                    "device_map"
                ] = {
                    "": 0
                }

            else:

                model_kwargs[
                    "device_map"
                ] = "auto"


    else:

        model_kwargs[
            "torch_dtype"
        ] = torch.float32


    model = (
        AutoModelForCausalLM
        .from_pretrained(
            str(checkpoint),
            **model_kwargs,
        )
    )


    # CPU only.
    if not torch.cuda.is_available():

        model = model.to(
            DEVICE
        )


    model.eval()


    print(
        f"Model loaded successfully: "
        f"{model_name}"
    )

    print()

    return (
        tokenizer,
        model,
    )


# ============================================================
# PROMPT BUILDERS
# ============================================================

SYSTEM_PROMPT = (

    "You are a helpful assistant that "
    "generates natural Hindi-English "
    "code-mixed (Hinglish) text. "

    "Follow the user's instruction exactly. "

    "Use natural Roman-script Hinglish "
    "unless the prompt specifies otherwise. "

    "Do not explain your answer. "

    "Return only the requested response."
)


def build_chat_prompt(
    tokenizer,
    user_prompt,
):

    messages = [

        {
            "role":
                "system",

            "content":
                SYSTEM_PROMPT,
        },

        {
            "role":
                "user",

            "content":
                user_prompt,
        },
    ]


    if getattr(
        tokenizer,
        "chat_template",
        None,
    ):

        return (
            tokenizer
            .apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )


    return (

        f"System: "
        f"{SYSTEM_PROMPT}\n\n"

        f"User: "
        f"{user_prompt}\n\n"

        "Assistant:"
    )


def build_hinggpt_prompt(
    user_prompt,
):

    return (

        "Generate a natural Hinglish "
        "response for the following "
        "instruction.\n"

        "Follow the instruction exactly.\n"

        "Use Roman-script Hindi mixed "
        "naturally with English.\n"

        "Return only the response.\n\n"

        f"Instruction: "
        f"{user_prompt}\n\n"

        "Response:"
    )


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize_prompt(
    tokenizer,
    prompt,
):

    encoded = tokenizer(

        prompt,

        return_tensors="pt",

        truncation=True,

        max_length=
            MAX_INPUT_TOKENS,

        padding=False,
    )


    # IMPORTANT:
    #
    # Never pass BatchEncoding directly
    # to model.generate().
    #
    # Explicit tensors only.

    input_ids = (
        encoded["input_ids"]
    )

    attention_mask = (
        encoded.get(
            "attention_mask",
            torch.ones_like(
                input_ids
            ),
        )
    )


    # For normal single-GPU inference.
    if torch.cuda.is_available():

        input_ids = (
            input_ids.to(
                "cuda"
            )
        )

        attention_mask = (
            attention_mask.to(
                "cuda"
            )
        )


    return (
        input_ids,
        attention_mask,
    )


# ============================================================
# GENERATE ONE RESPONSE
# ============================================================

def generate_response(
    tokenizer,
    model,
    prompt,
    model_type,
    model_name,
):

    try:

        # ----------------------------------------------------
        # Build prompt
        # ----------------------------------------------------

        if model_type == "chat":

            formatted_prompt = (
                build_chat_prompt(
                    tokenizer,
                    prompt,
                )
            )

        else:

            formatted_prompt = (
                build_hinggpt_prompt(
                    prompt,
                )
            )


        # ----------------------------------------------------
        # Tokenize
        # ----------------------------------------------------

        (
            input_ids,
            attention_mask,
        ) = tokenize_prompt(
            tokenizer,
            formatted_prompt,
        )


        input_length = (
            input_ids.shape[1]
        )


        # ====================================================
        # QWEN 2.5 7B
        #
        # GREEDY GENERATION
        # ====================================================

        if model_name == "Qwen2.5-7B":

            with torch.inference_mode():

                generated = (
                    model.generate(

                        input_ids=
                            input_ids,

                        attention_mask=
                            attention_mask,

                        max_new_tokens=
                            MAX_NEW_TOKENS,

                        do_sample=False,

                        num_beams=1,

                        repetition_penalty=
                            REPETITION_PENALTY,

                        pad_token_id=
                            tokenizer.pad_token_id,

                        eos_token_id=
                            tokenizer.eos_token_id,

                        use_cache=True,
                    )
                )


        # ====================================================
        # OTHER MODELS
        #
        # CONTROLLED SAMPLING
        # ====================================================

        else:

            with torch.inference_mode():

                generated = (
                    model.generate(

                        input_ids=
                            input_ids,

                        attention_mask=
                            attention_mask,

                        max_new_tokens=
                            MAX_NEW_TOKENS,

                        do_sample=True,

                        temperature=
                            TEMPERATURE,

                        top_p=
                            TOP_P,

                        repetition_penalty=
                            REPETITION_PENALTY,

                        num_beams=1,

                        pad_token_id=
                            tokenizer.pad_token_id,

                        eos_token_id=
                            tokenizer.eos_token_id,

                        use_cache=True,
                    )
                )


        # ----------------------------------------------------
        # Decode ONLY generated tokens
        # ----------------------------------------------------

        new_tokens = (
            generated[
                0,
                input_length:
            ]
        )


        response = (
            tokenizer.decode(
                new_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            .strip()
        )


        if not response:

            return (
                "[GENERATION_ERROR] "
                "empty response"
            )


        return response


    except Exception as e:

        return (
            "[GENERATION_ERROR] "
            + repr(e)
        )


# ============================================================
# QUALITY CHECK
# ============================================================

def is_bad_generation(
    response
):

    if not response:
        return True


    if response.startswith(
        "[GENERATION_ERROR]"
    ):

        return True


    text = response.strip()


    alnum_count = sum(
        c.isalnum()
        for c in text
    )


    if (
        len(text) >= 10
        and alnum_count == 0
    ):

        return True


    return False


# ============================================================
# MODEL RESULT COUNT
# ============================================================

def get_model_rows(
    model_name
):

    return [

        r

        for r in results

        if str(
            r.get(
                "model",
                ""
            )
        ) == model_name
    ]


def get_valid_model_count(
    model_name
):

    rows = get_model_rows(
        model_name
    )

    return sum(

        1

        for r in rows

        if (

            str(
                r.get(
                    "response",
                    ""
                )
            ).strip()

            and not str(
                r.get(
                    "response",
                    ""
                )
            ).startswith(
                "[GENERATION_ERROR]"
            )
        )
    )


# ============================================================
# MAIN GENERATION LOOP
# ============================================================

for model_name, model_cfg in MODELS.items():

    print()
    print("#" * 80)
    print(
        f"MODEL: {model_name}"
    )
    print("#" * 80)
    print()


    tokenizer = None
    model = None


    try:

        # ----------------------------------------------------
        # Count existing rows
        # ----------------------------------------------------

        total = len(
            bench
        )


        already_completed = sum(

            1

            for row_index in range(
                total
            )

            if (
                model_name,
                row_index,
            ) in completed
        )


        print(
            f"Existing valid rows: "
            f"{already_completed}/{total}"
        )


        # ----------------------------------------------------
        # If complete, DO NOT load model.
        # ----------------------------------------------------

        if already_completed >= total:

            print(
                f"{model_name}: "
                "already complete. SKIPPING MODEL LOAD."
            )

            print()

            continue


        # ----------------------------------------------------
        # Load model
        # ----------------------------------------------------

        tokenizer, model = (
            load_model(
                model_name,
                model_cfg,
            )
        )


        # ----------------------------------------------------
        # Generate missing rows
        # ----------------------------------------------------

        for row_index, row in (
            bench.iterrows()
        ):

            key = (
                model_name,
                int(row_index),
            )


            # ------------------------------------------------
            # Resume:
            # Skip existing VALID result.
            # ------------------------------------------------

            if key in completed:

                print(
                    f"[{row_index + 1}/"
                    f"{total}] "
                    "SKIP existing"
                )

                continue


            user_prompt = str(
                row[
                    PROMPT_COLUMN
                ]
            )


            print(
                f"[{row_index + 1}/"
                f"{total}] "
                f"Generating..."
            )


            response = (
                generate_response(

                    tokenizer=
                        tokenizer,

                    model=
                        model,

                    prompt=
                        user_prompt,

                    model_type=
                        model_cfg["type"],

                    model_name=
                        model_name,
                )
            )


            # ------------------------------------------------
            # Print response status
            # ------------------------------------------------

            if is_bad_generation(
                response
            ):

                print(
                    "  WARNING: "
                    + response[:200]
                )

            else:

                preview = (
                    response
                    .replace(
                        "\n",
                        " "
                    )[:150]
                )

                print(
                    f"  OK: "
                    f"{preview}"
                )


            # ------------------------------------------------
            # Remove previous failed row
            # if it exists.
            # ------------------------------------------------

            results = [

                r

                for r in results

                if not (

                    str(
                        r.get(
                            "model",
                            ""
                        )
                    )
                    == model_name

                    and

                    normalize_row_index(
                        r.get(
                            "row_index",
                            None
                        )
                    )
                    == int(row_index)
                )
            ]


            # ------------------------------------------------
            # Add latest result
            # ------------------------------------------------

            results.append(

                {
                    "row_index":
                        int(row_index),

                    "model":
                        model_name,

                    "prompt":
                        user_prompt,

                    "response":
                        response,
                }
            )


            # ------------------------------------------------
            # Mark completed only if valid
            # ------------------------------------------------

            if not is_bad_generation(
                response
            ):

                completed.add(
                    key
                )


            # ------------------------------------------------
            # Save EVERY ROW
            # ------------------------------------------------

            save_results()


            if torch.cuda.is_available():

                torch.cuda.empty_cache()


        # ----------------------------------------------------
        # Model completion
        # ----------------------------------------------------

        valid_count = (
            get_valid_model_count(
                model_name
            )
        )


        print()
        print(
            f"{model_name} COMPLETE: "
            f"{valid_count}/{total} valid"
        )
        print()


    except Exception as e:

        print()
        print(
            "#" * 80
        )

        print(
            f"MODEL FAILED: "
            f"{model_name}"
        )

        print(
            repr(e)
        )

        print(
            "#" * 80
        )

        print()


        # IMPORTANT:
        # Do not delete existing results.
        save_results()


    finally:

        # Safe cleanup
        try:

            if model is not None:

                del model

        except Exception:
            pass


        try:

            if tokenizer is not None:

                del tokenizer

        except Exception:
            pass


        if torch.cuda.is_available():

            torch.cuda.empty_cache()


# ============================================================
# FINAL SAVE
# ============================================================

save_results()


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 80)
print("V3 GENERATION COMPLETE")
print("=" * 80)

print(
    f"CSV   : {CSV_PATH}"
)

print(
    f"JSONL : {JSONL_PATH}"
)

print(
    f"CONFIG: {CONFIG_PATH}"
)

print()


if CSV_PATH.exists():

    final_df = pd.read_csv(
        CSV_PATH
    )


    # --------------------------------------------------------
    # Remove invalid NaN model rows if any.
    # --------------------------------------------------------

    final_df = final_df[
        final_df["model"]
        .notna()
    ].copy()


    print(
        f"Total rows: "
        f"{len(final_df)}"
    )

    print()


    print(
        "MODEL COUNTS:"
    )

    counts = (
        final_df[
            "model"
        ]
        .value_counts()
    )

    for model_name in MODELS:

        print(
            f"{model_name:20s} "
            f"{int(counts.get(model_name, 0))}"
        )


    print()


    print(
        "VALID RESPONSES:"
    )


    all_complete = True


    for model_name in MODELS:

        subset = final_df[
            final_df["model"]
            == model_name
        ]


        valid = subset[
            ~subset[
                "response"
            ]
            .astype(str)
            .str.startswith(
                "[GENERATION_ERROR]"
            )
        ]


        valid_count = len(
            valid
        )

        total_count = len(
            subset
        )


        print(
            f"{model_name:20s} "
            f"{valid_count}/"
            f"{total_count}"
        )


        if (
            valid_count
            != len(bench)
        ):

            all_complete = False


    print()


    # --------------------------------------------------------
    # FINAL TARGET
    # --------------------------------------------------------

    if (
        len(final_df)
        == len(bench) * len(MODELS)
        and all_complete
    ):

        print(
            "=" * 80
        )

        print(
            "SUCCESS: ALL 4 MODELS HAVE "
            "34/34 VALID GENERATIONS"
        )

        print(
            "TOTAL VALID GENERATIONS: "
            "136/136"
        )

        print(
            "=" * 80
        )

        print()
        print(
            "NOW you can run the "
            "independent Mistral judge."
        )

    else:

        print(
            "=" * 80
        )

        print(
            "NOT COMPLETE YET"
        )

        print(
            "Do NOT run the LLM judge yet."
        )

        print(
            "Qwen2.5-7B must reach 34/34."
        )

        print(
            "=" * 80
        )


else:

    print(
        "ERROR: V3 CSV was not created."
    )

print()
