import os
import json
import time
import random
import numpy as np
import pandas as pd
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel


# ============================================================
# CONFIG
# ============================================================

BASE_MODEL = "microsoft/Phi-3.5-mini-instruct"

CHECKPOINT = r".\models\phi35_mini_cpt\checkpoint-17340"

BENCHMARK_FILE = r".\data\benchmarks\hinglish_bench_test.csv"

OUTPUT_DIR = r".\outputs\hinglish_bench"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_JSONL = os.path.join(
    OUTPUT_DIR,
    "phi35_mini_hinglish_bench_generations.jsonl"
)

OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    "phi35_mini_hinglish_bench_generations.csv"
)


# ============================================================
# LOCKED EVALUATION SETTINGS
# ============================================================

MAX_INPUT_LENGTH = 128
MAX_NEW_TOKENS = 128

TEMPERATURE = 0.7
TOP_P = 0.9
REPETITION_PENALTY = 1.1

SEED = 42


# ============================================================
# SEED
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    CHECKPOINT,
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Tokenizer loaded.")
print("Tokenizer vocab:", len(tokenizer))


# ============================================================
# 4-BIT QUANTIZATION
# ============================================================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)


# ============================================================
# LOAD BASE MODEL
# ============================================================

print("\nLoading Phi-3.5-mini base model...")

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)

print("Base model loaded successfully.")


# ============================================================
# VOCABULARY ALIGNMENT
# ============================================================

model_vocab = model.get_input_embeddings().weight.shape[0]
tokenizer_vocab = len(tokenizer)

print("\nModel vocabulary    :", model_vocab)
print("Tokenizer vocabulary:", tokenizer_vocab)

if model_vocab != tokenizer_vocab:
    print(
        f"Resizing embeddings: "
        f"{model_vocab} -> {tokenizer_vocab}"
    )

    model.resize_token_embeddings(tokenizer_vocab)


# ============================================================
# LOAD CPT LoRA
# ============================================================

print("\nLoading CPT LoRA checkpoint...")

model = PeftModel.from_pretrained(
    model,
    CHECKPOINT,
    is_trainable=False
)

model.eval()

# LOCKED setting
model.config.use_cache = False

print("\n" + "=" * 70)
print("✅ PHI-3.5-MINI + CPT LoRA LOADED SUCCESSFULLY")
print("=" * 70)


# ============================================================
# LOAD BENCHMARK
# ============================================================

df = pd.read_csv(BENCHMARK_FILE)

print("\nBenchmark rows:", len(df))
print("Columns:", list(df.columns))

assert len(df) == 34, (
    f"Expected 34 benchmark prompts, found {len(df)}"
)


# ============================================================
# GENERATION
# ============================================================

results = []

start_time = time.time()

# Remove old output so this is a fresh run
if os.path.exists(OUTPUT_JSONL):
    os.remove(OUTPUT_JSONL)

for i, row in df.iterrows():

    prompt = str(row["prompt"])

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LENGTH
    )

    inputs = {
        k: v.to(model.device)
        for k, v in inputs.items()
    }

    with torch.no_grad():

        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            use_cache=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # Remove prompt tokens
    generated_ids = output_ids[
        0
    ][
        inputs["input_ids"].shape[1]:
    ]

    response = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True
    ).strip()

    result = {
        "id": row["id"],
        "category": row["category"],
        "category_name": row["category_name"],
        "prompt": prompt,
        "expected_style": row["expected_style"],
        "language": row["language"],
        "split": row["split"],
        "response": response
    }

    results.append(result)

    # Save after every sample
    with open(
        OUTPUT_JSONL,
        "a",
        encoding="utf-8"
    ) as f:

        f.write(
            json.dumps(
                result,
                ensure_ascii=False
            ) + "\n"
        )

    elapsed = time.time() - start_time

    print(
        f"[{i + 1:02d}/{len(df)}] "
        f"{elapsed / 60:.2f} min | "
        f"{row['category_name']} | "
        f"{response[:100].replace(chr(10), ' ')}"
    )


# ============================================================
# SAVE CSV
# ============================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# FINAL
# ============================================================

total_time = time.time() - start_time

print("\n" + "=" * 70)
print("✅ PHI-3.5-MINI HINGLISH-BENCH GENERATION COMPLETE")
print("=" * 70)

print("Samples:", len(results_df))
print(f"Time: {total_time / 60:.2f} minutes")

print("\nJSONL:")
print(OUTPUT_JSONL)

print("\nCSV:")
print(OUTPUT_CSV)