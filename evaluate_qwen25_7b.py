import os

# Must be set BEFORE torch import
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

BASE_MODEL = "Qwen/Qwen2.5-7B"

CHECKPOINT = os.path.join(
    PROJECT_DIR,
    "models",
    "qwen25_7b_cpt",
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

OFFLOAD_DIR = os.path.join(
    PROJECT_DIR,
    "offload_qwen25_7b"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "qwen25_7b_final_evaluation.csv"
)

GENERATED_FILE = os.path.join(
    OUTPUT_DIR,
    "qwen25_7b_generated_samples.jsonl"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OFFLOAD_DIR, exist_ok=True)


# ============================================================
# LOCKED COMMON EVALUATION SETTINGS
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
print("QWEN2.5-7B FINAL EVALUATION")
print("=" * 70)

print("Base model :", BASE_MODEL)
print("Checkpoint :", CHECKPOINT)
print("Validation :", VALIDATION_FILE)
print("Offload    :", OFFLOAD_DIR)

if torch.cuda.is_available():
    print("GPU        :", torch.cuda.get_device_name(0))
    print(
        "VRAM       :",
        round(
            torch.cuda.get_device_properties(0).total_memory
            / 1024**3,
            2
        ),
        "GB"
    )


# ============================================================
# CHECK FILES
# ============================================================

if not os.path.isdir(CHECKPOINT):
    raise FileNotFoundError(
        f"Checkpoint not found:\n{CHECKPOINT}"
    )

if not os.path.isfile(VALIDATION_FILE):
    raise FileNotFoundError(
        f"Validation file not found:\n{VALIDATION_FILE}"
    )

print("\nCheckpoint verified.")
print("Validation file verified.")


# ============================================================
# LOAD VALIDATION
# ============================================================

print("\nLoading validation dataset...")

dataset = load_dataset(
    "json",
    data_files={
        "validation": VALIDATION_FILE
    }
)

val_dataset = dataset["validation"]

print("Validation samples:", len(val_dataset))

if len(val_dataset) != EXPECTED_VALIDATION:
    raise RuntimeError(
        f"Expected {EXPECTED_VALIDATION}, "
        f"found {len(val_dataset)}"
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

print("Tokenizer vocab:", len(tokenizer))
print("PAD token:", tokenizer.pad_token)
print("Padding:", tokenizer.padding_side)


# ============================================================
# 7B MODEL
# 8-BIT + CPU OFFLOAD
# ============================================================

print("\nLoading Qwen2.5-7B in 8-bit with CPU offload...")

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    max_memory={
        0: "4.8GiB",
        "cpu": "24GiB"
    },
    dtype=torch.float16,
    trust_remote_code=True,
    offload_folder=OFFLOAD_DIR,
    offload_state_dict=True,
    offload_buffers=True
)

print("Base model loaded.")


# ============================================================
# LOAD LoRA
# ============================================================

print("\nLoading LoRA checkpoint...")

model = PeftModel.from_pretrained(
    model,
    CHECKPOINT,
    is_trainable=False,
    device_map="auto",
    max_memory={
        0: "4.8GiB",
        "cpu": "24GiB"
    },
    offload_folder=OFFLOAD_DIR
)

model.eval()

# IMPORTANT: SAME FOR ALL 4 MODELS
model.config.use_cache = False

if hasattr(model, "generation_config"):
    model.generation_config.use_cache = False

print("LoRA loaded.")


# ============================================================
# DEVICE
# ============================================================

input_embedding_device = (
    model.get_input_embeddings()
    .weight.device
)

print(
    "Input embedding device:",
    input_embedding_device
)


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION LOSS + PERPLEXITY")
print("=" * 70)

texts = val_dataset["text"]

total_loss = 0.0
total_tokens = 0

validation_start = time.time()

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
    ].to(input_embedding_device)

    attention_mask = encoded[
        "attention_mask"
    ].to(input_embedding_device)

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

    if torch.cuda.is_available():
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

print("\nValidation Loss :", f"{validation_loss:.6f}")
print("Perplexity      :", f"{perplexity:.6f}")
print("Validation Time :", f"{validation_time:.2f} sec")


# ============================================================
# GENERATION
# ============================================================

print("\n" + "=" * 70)
print("GENERATION EVALUATION")
print("=" * 70)

generation_texts = texts[:EVAL_SAMPLES]

generated_outputs = []


# ============================================================
# CRASH-SAFE RESUME
# ============================================================

if os.path.isfile(GENERATED_FILE):

    print("\nExisting generation file found.")

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
                    "Existing generation file "
                    "is not contiguous."
                )

            generated_outputs.append(
                record["generated"]
            )

    print(
        f"Recovered: "
        f"{len(generated_outputs)}/"
        f"{EVAL_SAMPLES}"
    )

else:

    open(
        GENERATED_FILE,
        "w",
        encoding="utf-8"
    ).close()


# ============================================================
# GENERATION LOOP
# ============================================================

generation_start = time.time()

for start in range(
    len(generated_outputs),
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
    ].to(input_embedding_device)

    attention_mask = encoded[
        "attention_mask"
    ].to(input_embedding_device)

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

                # LOCKED
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

        # ----------------------------------------------------
        # SAVE IMMEDIATELY
        # ----------------------------------------------------

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

        elapsed = (
            time.time() - generation_start
        )

        print(
            f"Generated: "
            f"{batch_end}/{EVAL_SAMPLES} "
            f"| Time: {elapsed:.1f}s"
        )

    except Exception as e:

        print("\n" + "=" * 70)
        print("GENERATION ERROR")
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
            "\nRestart Python and rerun "
            "the same script."
        )

        raise

    finally:

        for name in [
            "encoded",
            "input_ids",
            "attention_mask",
            "generated",
            "new_tokens",
            "decoded"
        ]:
            if name in locals():
                try:
                    del globals()[name]
                except Exception:
                    pass

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


generation_time = (
    time.time() - generation_start
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
    len(set(all_tokens)) / len(all_tokens)
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

total_generated_tokens = 0
total_repeated_tokens = 0

for text in generated_outputs:

    tokens = text.strip().split()

    total_generated_tokens += len(tokens)

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

    total_repeated_tokens += repeated


repetition_rate = (
    total_repeated_tokens
    / total_generated_tokens
    if total_generated_tokens
    else 0.0
)


# ============================================================
# SAVE FINAL CSV
# ============================================================

results = {

    "Model":
        "Qwen2.5-7B",

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
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("QWEN2.5-7B FINAL RESULTS")
print("=" * 70)

for key in [
    "Validation Loss",
    "Perplexity",
    "Distinct-1",
    "Distinct-2",
    "Repetition Rate",
    "Generation Time"
]:
    print(
        f"{key:18s}: {results[key]}"
    )

print("\nCSV:")
print(OUTPUT_FILE)

print("\nJSONL:")
print(GENERATED_FILE)

print("\nQwen2.5-7B evaluation COMPLETE.")

# ============================================================
# QWEN2.5-7B — RTX 4050 GPU ONLY
# ============================================================

print("\nLoading Qwen2.5-7B in 4-bit NF4 on RTX 4050...")

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

print("✅ 7B base model loaded entirely on GPU.")

print("\nLoading LoRA checkpoint...")

model = PeftModel.from_pretrained(
    model,
    CHECKPOINT,
    is_trainable=False
)

model.eval()

# SAME evaluation method as the other models
model.config.use_cache = False

if hasattr(model, "generation_config"):
    model.generation_config.use_cache = False

print("✅ LoRA loaded.")

MODEL_DEVICE = torch.device("cuda:0")

print("🔥 Model device:", MODEL_DEVICE)