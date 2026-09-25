import os

# Set before importing torch
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True"
)

import math
import json
import time
import csv
import random

import torch

from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from peft import PeftModel


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

BASE_MODEL = "Qwen/Qwen2.5-3B"

CHECKPOINT = os.path.join(
    PROJECT_DIR,
    "models",
    "qwen25_3b_cpt",
    "checkpoint-17340"
)

VALIDATION_FILE = os.path.join(
    PROJECT_DIR,
    "data",
    "processed",
    "cpt",
    "validation.jsonl"
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "outputs"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "qwen25_3b_final_evaluation.csv"
)

GENERATED_FILE = os.path.join(
    OUTPUT_DIR,
    "qwen25_3b_generated_samples.jsonl"
)

PROGRESS_FILE = os.path.join(
    OUTPUT_DIR,
    "qwen25_3b_generation_progress.json"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOCKED EVALUATION SETTINGS
# ============================================================

MAX_LENGTH = 128
MAX_NEW_TOKENS = 50

EVAL_SAMPLES = 1000
BATCH_SIZE = 4

TEMPERATURE = 0.7
TOP_P = 0.9
REPETITION_PENALTY = 1.1

SEED = 42

EXPECTED_VALIDATION = 14601


# ============================================================
# SEED
# ============================================================

random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("QWEN2.5-3B - FINAL EVALUATION")
print("=" * 70)

print("Base model :", BASE_MODEL)
print("Checkpoint :", CHECKPOINT)
print("Validation :", VALIDATION_FILE)

if torch.cuda.is_available():
    print(
        "GPU        :",
        torch.cuda.get_device_name(0)
    )

    print(
        "VRAM       :",
        round(
            torch.cuda.get_device_properties(0)
            .total_memory / 1024**3,
            2
        ),
        "GB"
    )


# ============================================================
# FILE CHECKS
# ============================================================

if not os.path.isdir(CHECKPOINT):
    raise FileNotFoundError(
        f"Checkpoint not found:\n{CHECKPOINT}"
    )

if not os.path.isfile(VALIDATION_FILE):
    raise FileNotFoundError(
        f"Validation file not found:\n{VALIDATION_FILE}"
    )


# ============================================================
# VALIDATION DATA
# ============================================================

print("\nLoading validation dataset...")

dataset = load_dataset(
    "json",
    data_files={
        "validation": VALIDATION_FILE
    }
)

val_dataset = dataset["validation"]

print(
    "Validation samples:",
    len(val_dataset)
)

if len(val_dataset) != EXPECTED_VALIDATION:
    raise RuntimeError(
        f"Expected {EXPECTED_VALIDATION}, "
        f"found {len(val_dataset)}"
    )

if val_dataset.column_names != ["text"]:
    raise RuntimeError(
        f"Expected ['text'], "
        f"found {val_dataset.column_names}"
    )

print("Validation dataset verified.")


# ============================================================
# TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True
)

tokenizer.padding_side = "left"

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(
    "Tokenizer vocab:",
    len(tokenizer)
)

print(
    "PAD token:",
    tokenizer.pad_token
)

print(
    "Padding:",
    tokenizer.padding_side
)


# ============================================================
# LOAD 3B BASE MODEL
# ============================================================

print(
    "\nLoading Qwen2.5-3B "
    "in 4-bit NF4..."
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map={"": 0},
    dtype=torch.float16,
    trust_remote_code=True
)

print("Base model loaded.")


# ============================================================
# LOAD LoRA
# ============================================================

print("\nLoading LoRA checkpoint...")

model = PeftModel.from_pretrained(
    model,
    CHECKPOINT,
    is_trainable=False
)

model.eval()

model.config.use_cache = False

if hasattr(model, "generation_config"):
    model.generation_config.use_cache = False

print("LoRA loaded.")


# ============================================================
# DEVICE
# ============================================================

MODEL_DEVICE = next(
    model.parameters()
).device

print(
    "Model device:",
    MODEL_DEVICE
)


# ============================================================
# VALIDATION LOSS
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION LOSS + PERPLEXITY")
print("=" * 70)

total_loss = 0.0
total_tokens = 0

validation_start = time.time()

texts = val_dataset["text"]

for start in range(
    0,
    len(texts),
    BATCH_SIZE
):

    batch_texts = texts[
        start:start + BATCH_SIZE
    ]

    encoded = tokenizer(
        batch_texts,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
        return_tensors="pt"
    )

    input_ids = encoded[
        "input_ids"
    ].to(MODEL_DEVICE)

    attention_mask = encoded[
        "attention_mask"
    ].to(MODEL_DEVICE)

    labels = input_ids.clone()

    labels[
        attention_mask == 0
    ] = -100

    valid_tokens = (
        labels[:, 1:] != -100
    ).sum().item()

    if valid_tokens == 0:
        continue

    with torch.inference_mode():

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False
        )

    batch_loss = outputs.loss.item()

    total_loss += (
        batch_loss * valid_tokens
    )

    total_tokens += valid_tokens

    done = min(
        start + BATCH_SIZE,
        len(texts)
    )

    if (
        done % 500 < BATCH_SIZE
        or done == len(texts)
    ):
        print(
            f"Validation: "
            f"{done}/{len(texts)}"
        )

    del (
        encoded,
        input_ids,
        attention_mask,
        labels,
        outputs
    )

    torch.cuda.empty_cache()


validation_loss = (
    total_loss / total_tokens
)

perplexity = math.exp(
    validation_loss
)

validation_time = (
    time.time() - validation_start
)

print(
    f"\nValidation Loss : "
    f"{validation_loss:.6f}"
)

print(
    f"Perplexity      : "
    f"{perplexity:.6f}"
)

print(
    f"Validation Time : "
    f"{validation_time:.2f} sec"
)


# ============================================================
# GENERATION
# ============================================================

print("\n" + "=" * 70)
print("GENERATION EVALUATION")
print("=" * 70)

generation_texts = texts[
    :EVAL_SAMPLES
]


# ------------------------------------------------------------
# RESUME SUPPORT
# ------------------------------------------------------------

generated_outputs = []

resume_index = 0

if os.path.isfile(GENERATED_FILE):

    print(
        "\nExisting generation file found."
    )

    with open(
        GENERATED_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            record = json.loads(line)

            expected_id = len(
                generated_outputs
            )

            if record["sample_id"] != expected_id:
                raise RuntimeError(
                    "Generation file is not contiguous."
                )

            generated_outputs.append(
                record["generated"]
            )

    resume_index = len(
        generated_outputs
    )

    print(
        f"Recovered: "
        f"{resume_index}/{EVAL_SAMPLES}"
    )

else:

    open(
        GENERATED_FILE,
        "w",
        encoding="utf-8"
    ).close()


generation_start = time.time()


# ============================================================
# GENERATION LOOP
# ============================================================

for start in range(
    resume_index,
    EVAL_SAMPLES,
    BATCH_SIZE
):

    batch_end = min(
        start + BATCH_SIZE,
        EVAL_SAMPLES
    )

    print(
        f"\nGenerating "
        f"{start + 1}-{batch_end}/"
        f"{EVAL_SAMPLES}"
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    batch_texts = generation_texts[
        start:batch_end
    ]

    encoded = tokenizer(
        batch_texts,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
        return_tensors="pt"
    )

    input_ids = encoded[
        "input_ids"
    ].to(MODEL_DEVICE)

    attention_mask = encoded[
        "attention_mask"
    ].to(MODEL_DEVICE)

    try:

        with torch.inference_mode():

            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                repetition_penalty=REPETITION_PENALTY,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=False
            )

        input_length = input_ids.shape[1]

        new_tokens = generated[
            :,
            input_length:
        ]

        decoded = tokenizer.batch_decode(
            new_tokens,
            skip_special_tokens=True
        )

        # SAVE IMMEDIATELY
        with open(
            GENERATED_FILE,
            "a",
            encoding="utf-8"
        ) as f:

            for i, output in enumerate(decoded):

                sample_id = start + i

                record = {
                    "sample_id": sample_id,
                    "input": generation_texts[sample_id],
                    "generated": output
                }

                f.write(
                    json.dumps(
                        record,
                        ensure_ascii=False
                    ) + "\n"
                )

        generated_outputs.extend(
            decoded
        )

        with open(
            PROGRESS_FILE,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                {
                    "next_index": batch_end,
                    "seed": SEED
                },
                f
            )

        elapsed = (
            time.time()
            - generation_start
        )

        print(
            f"Generated: "
            f"{batch_end}/{EVAL_SAMPLES} "
            f"| Time: {elapsed:.1f}s"
        )

    except Exception as e:

        print("\n" + "=" * 70)
        print("GENERATION STOPPED")
        print("=" * 70)

        print(
            "Already saved:",
            len(generated_outputs)
        )

        print(
            "Error:",
            type(e).__name__,
            str(e)
        )

        print(
            "\nRestart Python and run "
            "the same script again."
        )

        raise

    finally:

        try:
            del (
                encoded,
                input_ids,
                attention_mask,
                generated,
                new_tokens,
                decoded
            )
        except Exception:
            pass

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


generation_time = (
    time.time()
    - generation_start
)


# ============================================================
# VERIFY
# ============================================================

if len(generated_outputs) != EVAL_SAMPLES:

    raise RuntimeError(
        f"Only {len(generated_outputs)}/"
        f"{EVAL_SAMPLES} generated."
    )


# ============================================================
# DISTINCT-1
# ============================================================

all_tokens = []

for text in generated_outputs:

    all_tokens.extend(
        text.strip().split()
    )


distinct_1 = (
    len(set(all_tokens))
    / len(all_tokens)
    if all_tokens
    else 0.0
)


# ============================================================
# DISTINCT-2
# ============================================================

all_bigrams = []

for text in generated_outputs:

    tokens = text.strip().split()

    for i in range(
        len(tokens) - 1
    ):

        all_bigrams.append(
            (
                tokens[i],
                tokens[i + 1]
            )
        )


distinct_2 = (
    len(set(all_bigrams))
    / len(all_bigrams)
    if all_bigrams
    else 0.0
)


# ============================================================
# REPETITION RATE
# ============================================================

total_tokens_generated = 0
total_repeated = 0

for text in generated_outputs:

    tokens = text.strip().split()

    total_tokens_generated += len(tokens)

    counts = {}

    for token in tokens:

        counts[token] = (
            counts.get(token, 0) + 1
        )

    repeated = sum(
        count - 1
        for count in counts.values()
        if count > 1
    )

    total_repeated += repeated


repetition_rate = (
    total_repeated
    / total_tokens_generated
    if total_tokens_generated
    else 0.0
)


# ============================================================
# SAVE CSV
# ============================================================

results = {

    "Model":
        "Qwen2.5-3B",

    "Validation Samples":
        len(val_dataset),

    "Generation Samples":
        len(generated_outputs),

    "Max Length":
        MAX_LENGTH,

    "Max New Tokens":
        MAX_NEW_TOKENS,

    "Batch Size":
        BATCH_SIZE,

    "Temperature":
        TEMPERATURE,

    "Top-p":
        TOP_P,

    "Repetition Penalty":
        REPETITION_PENALTY,

    "Validation Loss":
        validation_loss,

    "Perplexity":
        perplexity,

    "Distinct-1":
        distinct_1,

    "Distinct-2":
        distinct_2,

    "Repetition Rate":
        repetition_rate,

    "Validation Time":
        validation_time,

    "Generation Time":
        generation_time
}


with open(
    OUTPUT_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=results.keys()
    )

    writer.writeheader()
    writer.writerow(results)


# ============================================================
# CLEAN PROGRESS
# ============================================================

if os.path.isfile(PROGRESS_FILE):

    os.remove(
        PROGRESS_FILE
    )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("QWEN2.5-3B FINAL RESULTS")
print("=" * 70)

print(
    f"Validation Loss : "
    f"{validation_loss:.6f}"
)

print(
    f"Perplexity      : "
    f"{perplexity:.6f}"
)

print(
    f"Distinct-1      : "
    f"{distinct_1:.6f}"
)

print(
    f"Distinct-2      : "
    f"{distinct_2:.6f}"
)

print(
    f"Repetition Rate : "
    f"{repetition_rate:.6f}"
)

print(
    f"Generation Time : "
    f"{generation_time:.2f} sec"
)

print("\nCSV:")
print(OUTPUT_FILE)

print("\nJSONL:")
print(GENERATED_FILE)

print("\nQwen2.5-3B evaluation COMPLETE.")