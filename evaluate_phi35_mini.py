import os
import math
import time
import json
import torch
import pandas as pd

from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

from peft import PeftModel


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

BASE_MODEL = "microsoft/Phi-3.5-mini-instruct"

CHECKPOINT = os.path.join(
    PROJECT_DIR,
    "models",
    "phi35_mini_cpt",
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
    "phi35_mini_final_evaluation.csv"
)

GENERATED_FILE = os.path.join(
    OUTPUT_DIR,
    "phi35_mini_generated_samples.jsonl"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# EVALUATION SETTINGS
# EXACT SAME FOR ALL 4 MODELS
# ============================================================

MAX_LENGTH = 128
MAX_NEW_TOKENS = 50
EVAL_SAMPLES = 1000
BATCH_SIZE = 4

TEMPERATURE = 0.7
TOP_P = 0.9
REPETITION_PENALTY = 1.1

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("PHI-3.5-MINI - FINAL EVALUATION")
print("=" * 70)

print("Device      :", DEVICE)

if torch.cuda.is_available():

    print(
        "GPU         :",
        torch.cuda.get_device_name(0)
    )

    print(
        "VRAM        :",
        round(
            torch.cuda.get_device_properties(0)
            .total_memory
            / 1024**3,
            2
        ),
        "GB"
    )

print(
    "Base model  :",
    BASE_MODEL
)

print(
    "Checkpoint  :",
    CHECKPOINT
)

print(
    "Validation  :",
    VALIDATION_FILE
)

print(
    "Output      :",
    OUTPUT_FILE
)


# ============================================================
# FILE CHECKS
# ============================================================

if not os.path.isfile(
    VALIDATION_FILE
):

    raise FileNotFoundError(
        "Validation file not found:\n"
        + VALIDATION_FILE
    )


if not os.path.isdir(
    CHECKPOINT
):

    raise FileNotFoundError(
        "Checkpoint not found:\n"
        + CHECKPOINT
    )


print(
    "\nRequired files found"
)


# ============================================================
# LOAD VALIDATION DATASET
# ============================================================

print(
    "\nLoading validation dataset..."
)

dataset = load_dataset(
    "json",
    data_files={
        "validation":
            VALIDATION_FILE
    }
)

val_dataset = dataset[
    "validation"
]

print(
    "Validation samples:",
    len(val_dataset)
)

print(
    "Columns:",
    val_dataset.column_names
)

if len(val_dataset) != 14601:

    raise RuntimeError(
        "Expected 14601 validation rows, "
        f"found {len(val_dataset)}"
    )

if val_dataset.column_names != [
    "text"
]:

    raise RuntimeError(
        "Expected ['text'] column, "
        f"found {val_dataset.column_names}"
    )

print(
    "Validation dataset verified"
)


# ============================================================
# LOAD TOKENIZER
# ============================================================

print(
    "\nLoading Phi tokenizer..."
)

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True
)

# Decoder-only model
tokenizer.padding_side = "left"

if tokenizer.pad_token is None:

    tokenizer.pad_token = (
        tokenizer.eos_token
    )

print(
    "Tokenizer vocab size:",
    len(tokenizer)
)

print(
    "PAD token:",
    tokenizer.pad_token
)

print(
    "PAD token ID:",
    tokenizer.pad_token_id
)

print(
    "Padding side:",
    tokenizer.padding_side
)


# ============================================================
# 4-BIT NF4 QUANTIZATION
# ============================================================

print(
    "\nPreparing 4-bit NF4 quantization..."
)

bnb_config = BitsAndBytesConfig(

    load_in_4bit=True,

    bnb_4bit_quant_type="nf4",

    bnb_4bit_use_double_quant=True,

    bnb_4bit_compute_dtype=torch.float16
)


# ============================================================
# LOAD BASE MODEL
# ============================================================

print(
    "\nLoading Phi-3.5-mini base model..."
)

model = AutoModelForCausalLM.from_pretrained(

    BASE_MODEL,

    quantization_config=bnb_config,

    device_map="auto",

    torch_dtype=torch.float16,

    trust_remote_code=True
)


# ============================================================
# VOCABULARY COMPATIBILITY
# ============================================================

model_vocab_size = (
    model
    .get_input_embeddings()
    .weight
    .shape[0]
)

tokenizer_vocab_size = len(
    tokenizer
)

print(
    "Model vocab size:",
    model_vocab_size
)

print(
    "Tokenizer vocab size:",
    tokenizer_vocab_size
)


if model_vocab_size != tokenizer_vocab_size:

    print(
        "Vocabulary mismatch detected."
    )

    print(
        "Resizing embeddings..."
    )

    model.resize_token_embeddings(
        tokenizer_vocab_size
    )

    model_vocab_size = (
        model
        .get_input_embeddings()
        .weight
        .shape[0]
    )

    print(
        "New model vocab size:",
        model_vocab_size
    )


if model_vocab_size != tokenizer_vocab_size:

    raise RuntimeError(
        "Vocabulary sizes still do not match!"
    )

print(
    "Vocabulary sizes match"
)


# ============================================================
# PAD TOKEN
# ============================================================

model.config.pad_token_id = (
    tokenizer.pad_token_id
)


# ============================================================
# PHI CACHE COMPATIBILITY
# ============================================================

# Current Transformers cache API can conflict
# with Phi-3.5 custom modeling code.
#
# Therefore cache is disabled for Phi only.
#
# Evaluation dataset and generation parameters
# remain exactly the same as the other models.

model.config.use_cache = False

if hasattr(
    model,
    "generation_config"
):

    model.generation_config.use_cache = False


# ============================================================
# LOAD LoRA CHECKPOINT
# ============================================================

print(
    "\nLoading Phi CPT LoRA checkpoint..."
)

model = PeftModel.from_pretrained(
    model,
    CHECKPOINT
)

model.eval()

model.config.pad_token_id = (
    tokenizer.pad_token_id
)

model.config.use_cache = False

if hasattr(
    model,
    "generation_config"
):

    model.generation_config.pad_token_id = (
        tokenizer.pad_token_id
    )

    model.generation_config.use_cache = False


print(
    "Phi-3.5-mini loaded successfully"
)


# ============================================================
# VALIDATION LOSS + PERPLEXITY
# ============================================================

def calculate_loss_and_ppl(
    model,
    tokenizer,
    dataset
):

    print(
        "\n" + "=" * 70
    )

    print(
        "CALCULATING VALIDATION LOSS"
    )

    print(
        "=" * 70
    )

    total_loss = 0.0

    total_tokens = 0

    start_time = time.time()

    model.eval()

    for start in range(
        0,
        len(dataset),
        BATCH_SIZE
    ):

        end = min(
            start + BATCH_SIZE,
            len(dataset)
        )

        batch = dataset[
            start:end
        ]

        texts = batch[
            "text"
        ]

        encoded = tokenizer(

            texts,

            padding=True,

            truncation=True,

            max_length=MAX_LENGTH,

            return_tensors="pt"
        )

        input_ids = encoded[
            "input_ids"
        ].to(DEVICE)

        attention_mask = encoded[
            "attention_mask"
        ].to(DEVICE)

        labels = input_ids.clone()

        labels[
            attention_mask == 0
        ] = -100

        with torch.inference_mode():

            outputs = model(

                input_ids=input_ids,

                attention_mask=attention_mask,

                labels=labels,

                use_cache=False
            )

        loss = outputs.loss

        # Causal LM loss predicts
        # token t+1, so first position
        # is excluded from token count.

        valid_tokens = (
            attention_mask[:, 1:]
            .sum()
            .item()
        )

        total_loss += (
            loss.item()
            * valid_tokens
        )

        total_tokens += (
            valid_tokens
        )

        if (
            end % 1000 == 0
            or end == len(dataset)
        ):

            print(
                f"Processed: "
                f"{end}/{len(dataset)}"
            )

    avg_loss = (
        total_loss
        / total_tokens
    )

    perplexity = math.exp(
        avg_loss
    )

    elapsed = (
        time.time()
        - start_time
    )

    return (
        avg_loss,
        perplexity,
        elapsed
    )


# ============================================================
# DISTINCT-N
# ============================================================

def distinct_n(
    texts,
    n
):

    total_ngrams = 0

    unique_ngrams = set()

    for text in texts:

        tokens = (
            text
            .lower()
            .split()
        )

        if len(tokens) < n:

            continue

        ngrams = [

            tuple(
                tokens[i:i+n]
            )

            for i in range(
                len(tokens) - n + 1
            )
        ]

        total_ngrams += len(
            ngrams
        )

        unique_ngrams.update(
            ngrams
        )

    if total_ngrams == 0:

        return 0.0

    return (
        len(unique_ngrams)
        / total_ngrams
    )


# ============================================================
# REPETITION RATE
# ============================================================

def repetition_rate(
    texts
):

    rates = []

    for text in texts:

        tokens = (
            text
            .lower()
            .split()
        )

        if not tokens:

            continue

        repeated_tokens = (
            len(tokens)
            - len(set(tokens))
        )

        rates.append(
            repeated_tokens
            / len(tokens)
        )

    if not rates:

        return 0.0

    return (
        sum(rates)
        / len(rates)
    )


# ============================================================
# GENERATION EVALUATION
# ============================================================

def generation_evaluation(
    model,
    tokenizer,
    dataset
):

    sample_count = min(
        EVAL_SAMPLES,
        len(dataset)
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "GENERATION EVALUATION "
        f"({sample_count} samples)"
    )

    print(
        "=" * 70
    )

    generated_texts = []

    start_time = time.time()

    model.eval()

    for start in range(
        0,
        sample_count,
        BATCH_SIZE
    ):

        end = min(
            start + BATCH_SIZE,
            sample_count
        )

        batch = dataset[
            start:end
        ]

        prompts = batch[
            "text"
        ]

        encoded = tokenizer(

            prompts,

            padding=True,

            truncation=True,

            max_length=MAX_LENGTH,

            return_tensors="pt"
        )

        input_ids = encoded[
            "input_ids"
        ].to(DEVICE)

        attention_mask = encoded[
            "attention_mask"
        ].to(DEVICE)

        input_width = (
            input_ids.shape[1]
        )

        with torch.inference_mode():

            outputs = model.generate(

                input_ids=input_ids,

                attention_mask=attention_mask,

                max_new_tokens=MAX_NEW_TOKENS,

                do_sample=True,

                temperature=TEMPERATURE,

                top_p=TOP_P,

                repetition_penalty=(
                    REPETITION_PENALTY
                ),

                pad_token_id=(
                    tokenizer.pad_token_id
                ),

                eos_token_id=(
                    tokenizer.eos_token_id
                ),

                use_cache=False
            )

        generated_tokens = (
            outputs[:, input_width:]
        )

        for i in range(
            len(prompts)
        ):

            generated_text = (
                tokenizer.decode(

                    generated_tokens[i],

                    skip_special_tokens=True

                )
                .strip()
            )

            generated_texts.append(
                generated_text
            )

        if (
            end % 100 == 0
            or end == sample_count
        ):

            elapsed = (
                time.time()
                - start_time
            )

            print(
                f"Generated: "
                f"{end}/{sample_count} "
                f"| Time: "
                f"{elapsed:.1f}s"
            )

    generation_time = (
        time.time()
        - start_time
    )

    d1 = distinct_n(
        generated_texts,
        1
    )

    d2 = distinct_n(
        generated_texts,
        2
    )

    rep = repetition_rate(
        generated_texts
    )

    return (
        d1,
        d2,
        rep,
        generation_time,
        generated_texts
    )


# ============================================================
# RUN VALIDATION
# ============================================================

(
    val_loss,
    perplexity,
    ppl_time
) = calculate_loss_and_ppl(

    model,

    tokenizer,

    val_dataset
)


print(
    "\nValidation Loss :",
    f"{val_loss:.6f}"
)

print(
    "Perplexity      :",
    f"{perplexity:.6f}"
)

print(
    "PPL Time        :",
    f"{ppl_time:.2f} sec"
)


# ============================================================
# RUN GENERATION
# ============================================================

(
    distinct1,
    distinct2,
    repetition,
    generation_time,
    generated_texts
) = generation_evaluation(

    model,

    tokenizer,

    val_dataset
)


# ============================================================
# FINAL RESULTS
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "PHI-3.5-MINI FINAL RESULTS"
)

print(
    "=" * 70
)

print(
    f"Validation Loss  : "
    f"{val_loss:.6f}"
)

print(
    f"Perplexity       : "
    f"{perplexity:.6f}"
)

print(
    f"Distinct-1       : "
    f"{distinct1:.6f}"
)

print(
    f"Distinct-2       : "
    f"{distinct2:.6f}"
)

print(
    f"Repetition Rate  : "
    f"{repetition:.6f}"
)

print(
    f"Generation Time  : "
    f"{generation_time:.2f} sec"
)


# ============================================================
# SAVE GENERATED SAMPLES
# ============================================================

print(
    "\nSaving generated samples..."
)

with open(
    GENERATED_FILE,
    "w",
    encoding="utf-8"
) as f:

    for i, text in enumerate(
        generated_texts
    ):

        record = {

            "sample_id":
                i + 1,

            "generated_text":
                text
        }

        f.write(
            json.dumps(
                record,
                ensure_ascii=False
            )
            + "\n"
        )

print(
    "Generated samples saved:",
    GENERATED_FILE
)


# ============================================================
# SAVE CSV
# ============================================================

results = {

    "Model":
        "Phi-3.5-mini",

    "Validation Samples":
        len(val_dataset),

    "Generation Samples":
        len(generated_texts),

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
        val_loss,

    "Perplexity":
        perplexity,

    "Distinct-1":
        distinct1,

    "Distinct-2":
        distinct2,

    "Repetition Rate":
        repetition,

    "Generation Time":
        generation_time
}

results_df = pd.DataFrame(
    [results]
)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# CONFIRMATION
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "RESULT SAVED"
)

print(
    "=" * 70
)

print(
    "CSV:",
    OUTPUT_FILE
)

print(
    "JSONL:",
    GENERATED_FILE
)

print(
    "\nFinal CSV content:\n"
)

print(
    results_df.to_string(
        index=False
    )
)

print(
    "\nPhi-3.5-mini evaluation "
    "completed successfully!"
)