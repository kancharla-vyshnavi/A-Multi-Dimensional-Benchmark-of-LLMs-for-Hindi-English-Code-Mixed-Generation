import os
import math
import time

import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

BASE_MODEL = "l3cube-pune/hing-gpt"

CHECKPOINT = os.path.join(
    PROJECT_DIR,
    "models",
    "hinggpt_cpt",
    "checkpoint-17340"
)

VALIDATION_FILE = os.path.join(
    PROJECT_DIR,
    "data",
    "processed",
    "cpt",
    "validation.jsonl"
)

OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "hinggpt_final_evaluation.csv"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# EVALUATION SETTINGS
# ============================================================

MAX_LENGTH = 128
MAX_NEW_TOKENS = 50
EVAL_SAMPLES = 1000
BATCH_SIZE = 4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("HINGGPT - FINAL EVALUATION")
print("=" * 70)

print("Device      :", DEVICE)

if torch.cuda.is_available():
    print("GPU         :", torch.cuda.get_device_name(0))
    print(
        "VRAM        :",
        round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
        "GB"
    )

print("Base model  :", BASE_MODEL)
print("Checkpoint  :", CHECKPOINT)
print("Validation  :", VALIDATION_FILE)
print("Output      :", OUTPUT_FILE)


# ============================================================
# FILE CHECKS
# ============================================================

if not os.path.isfile(VALIDATION_FILE):
    raise FileNotFoundError(
        f"Validation file not found:\n{VALIDATION_FILE}"
    )

if not os.path.isdir(CHECKPOINT):
    raise FileNotFoundError(
        f"Checkpoint not found:\n{CHECKPOINT}"
    )

print("\n✅ Required files found")


# ============================================================
# LOAD VALIDATION DATASET
# ============================================================

print("\nLoading validation dataset...")

dataset = load_dataset(
    "json",
    data_files={"validation": VALIDATION_FILE}
)

val_dataset = dataset["validation"]

print("Validation samples:", len(val_dataset))
print("Columns:", val_dataset.column_names)

if len(val_dataset) != 14601:
    raise RuntimeError(
        f"Expected 14601 validation rows, found {len(val_dataset)}"
    )

if val_dataset.column_names != ["text"]:
    raise RuntimeError(
        f"Expected ['text'] column, found {val_dataset.column_names}"
    )

print("✅ Validation dataset verified")


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("\nLoading HingGPT tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)

# Decoder-only model: left padding for batched generation
tokenizer.padding_side = "left"

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Tokenizer vocab size:", len(tokenizer))
print("PAD token:", tokenizer.pad_token)
print("PAD token ID:", tokenizer.pad_token_id)
print("Padding side:", tokenizer.padding_side)


# ============================================================
# LOAD BASE MODEL
# ============================================================

print("\nLoading base HingGPT model...")

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL
)

original_vocab_size = (
    base_model.get_input_embeddings().weight.shape[0]
)

print("Original base vocab size:", original_vocab_size)

# CPT checkpoint tokenizer may have a different vocabulary size.
base_model.resize_token_embeddings(len(tokenizer))

base_model.config.pad_token_id = tokenizer.pad_token_id

resized_vocab_size = (
    base_model.get_input_embeddings().weight.shape[0]
)

print("Resized base vocab size:", resized_vocab_size)
print("Tokenizer vocab size:", len(tokenizer))

if resized_vocab_size != len(tokenizer):
    raise RuntimeError("Vocabulary size mismatch!")

print("✅ Vocabulary sizes match")


# ============================================================
# LOAD LoRA CHECKPOINT
# ============================================================

print("\nLoading HingGPT CPT LoRA checkpoint...")

model = PeftModel.from_pretrained(
    base_model,
    CHECKPOINT
)

model = model.to(DEVICE)
model.eval()

model.config.pad_token_id = tokenizer.pad_token_id

print("✅ HingGPT loaded successfully")


# ============================================================
# VALIDATION LOSS + PERPLEXITY
# ============================================================

def calculate_loss_and_ppl(model, tokenizer, dataset):

    print("\n" + "=" * 70)
    print("CALCULATING VALIDATION LOSS")
    print("=" * 70)

    total_loss = 0.0
    total_tokens = 0

    start_time = time.time()

    model.eval()

    for start in range(0, len(dataset), BATCH_SIZE):

        end = min(
            start + BATCH_SIZE,
            len(dataset)
        )

        batch = dataset[start:end]
        texts = batch["text"]

        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        input_ids = encoded["input_ids"].to(DEVICE)
        attention_mask = encoded["attention_mask"].to(DEVICE)

        labels = input_ids.clone()

        labels[attention_mask == 0] = -100

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

        loss = outputs.loss

        # Number of non-padding tokens
        valid_tokens = attention_mask.sum().item()

        total_loss += loss.item() * valid_tokens
        total_tokens += valid_tokens

        if end % 1000 == 0 or end == len(dataset):
            print(
                f"Processed: {end}/{len(dataset)}"
            )

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    elapsed = time.time() - start_time

    return avg_loss, perplexity, elapsed


# ============================================================
# DISTINCT-N
# ============================================================

def distinct_n(texts, n):

    total_ngrams = 0
    unique_ngrams = set()

    for text in texts:

        tokens = text.lower().split()

        if len(tokens) < n:
            continue

        ngrams = [
            tuple(tokens[i:i+n])
            for i in range(len(tokens) - n + 1)
        ]

        total_ngrams += len(ngrams)
        unique_ngrams.update(ngrams)

    if total_ngrams == 0:
        return 0.0

    return len(unique_ngrams) / total_ngrams


# ============================================================
# REPETITION RATE
# ============================================================

def repetition_rate(texts):

    rates = []

    for text in texts:

        tokens = text.lower().split()

        if not tokens:
            continue

        repeated_tokens = (
            len(tokens) - len(set(tokens))
        )

        rates.append(
            repeated_tokens / len(tokens)
        )

    if not rates:
        return 0.0

    return sum(rates) / len(rates)


# ============================================================
# GENERATION EVALUATION
# ============================================================

def generation_evaluation(model, tokenizer, dataset):

    sample_count = min(
        EVAL_SAMPLES,
        len(dataset)
    )

    print("\n" + "=" * 70)
    print(
        f"GENERATION EVALUATION ({sample_count} samples)"
    )
    print("=" * 70)

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

        batch = dataset[start:end]
        prompts = batch["text"]

        encoded = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        input_ids = encoded["input_ids"].to(DEVICE)
        attention_mask = encoded["attention_mask"].to(DEVICE)

        input_width = input_ids.shape[1]

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        # With left padding, generated tokens start at
        # the padded input width.
        generated_tokens = outputs[:, input_width:]

        for i in range(len(prompts)):

            generated_text = tokenizer.decode(
                generated_tokens[i],
                skip_special_tokens=True
            ).strip()

            generated_texts.append(generated_text)

        if end % 100 == 0 or end == sample_count:

            elapsed = time.time() - start_time

            print(
                f"Generated: {end}/{sample_count} "
                f"| Time: {elapsed:.1f}s"
            )

    generation_time = time.time() - start_time

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
        generation_time
    )


# ============================================================
# RUN VALIDATION
# ============================================================

val_loss, perplexity, ppl_time = (
    calculate_loss_and_ppl(
        model,
        tokenizer,
        val_dataset
    )
)

print("\nValidation Loss :", f"{val_loss:.6f}")
print("Perplexity      :", f"{perplexity:.6f}")
print("PPL Time        :", f"{ppl_time:.2f} sec")


# ============================================================
# RUN GENERATION
# ============================================================

distinct1, distinct2, repetition, generation_time = (
    generation_evaluation(
        model,
        tokenizer,
        val_dataset
    )
)


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("HINGGPT FINAL RESULTS")
print("=" * 70)

print(
    f"Validation Loss  : {val_loss:.6f}"
)

print(
    f"Perplexity       : {perplexity:.6f}"
)

print(
    f"Distinct-1       : {distinct1:.6f}"
)

print(
    f"Distinct-2       : {distinct2:.6f}"
)

print(
    f"Repetition Rate  : {repetition:.6f}"
)

print(
    f"Generation Time  : {generation_time:.2f} sec"
)


# ============================================================
# SAVE CSV
# ============================================================

results = {
    "Model": "HingGPT",
    "Validation Loss": val_loss,
    "Perplexity": perplexity,
    "Distinct-1": distinct1,
    "Distinct-2": distinct2,
    "Repetition Rate": repetition,
    "Generation Time": generation_time
}

results_df = pd.DataFrame([results])

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# CONFIRMATION
# ============================================================

print("\n" + "=" * 70)
print("RESULT SAVED")
print("=" * 70)

print("File:", OUTPUT_FILE)

print("\nFinal CSV content:\n")
print(results_df.to_string(index=False))

print("\n✅ HingGPT evaluation completed successfully!")
