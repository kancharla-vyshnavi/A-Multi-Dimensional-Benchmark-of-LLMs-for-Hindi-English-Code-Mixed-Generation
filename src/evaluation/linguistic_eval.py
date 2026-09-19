import os
import re
import json
import torch
import pandas as pd
from collections import Counter
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
)

# ============================================================
# CONFIG
# ============================================================

PROJECT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
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
    "outputs",
    "linguistic_evaluation"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_NAME = "l3cube-pune/hing-bert-lid"

BATCH_SIZE = 32
MAX_LENGTH = 128

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("ROMAN HINGLISH LINGUISTIC EVALUATION")
print("=" * 70)

print(f"Validation file : {VALIDATION_FILE}")
print(f"LID model       : {MODEL_NAME}")
print(f"Device          : {DEVICE}")
print()


# ============================================================
# LOAD DATA
# ============================================================

if not os.path.exists(VALIDATION_FILE):
    raise FileNotFoundError(
        f"Validation file not found:\n{VALIDATION_FILE}"
    )

df = pd.read_json(
    VALIDATION_FILE,
    lines=True
)

if "text" not in df.columns:
    raise ValueError(
        "Validation file must contain a 'text' column."
    )

texts = (
    df["text"]
    .fillna("")
    .astype(str)
    .tolist()
)

print(f"Validation samples: {len(texts):,}")
print()


# ============================================================
# LOAD HINGBERT-LID
# ============================================================

print("Loading HingBERT-LID...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME
)

model.to(DEVICE)
model.eval()

print("HingBERT-LID loaded successfully.")
print()


# ============================================================
# LABEL INFORMATION
# ============================================================

print("Model labels:")

id2label = model.config.id2label

for idx, label in id2label.items():
    print(f"  {idx} -> {label}")

print()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_label(label):
    """
    Normalize model labels into:
        HI
        EN
        OTHER
    """

    label = str(label).upper()

    if "HIN" in label or label in {"HI", "HINDI"}:
        return "HI"

    if "ENG" in label or label in {"EN", "ENGLISH"}:
        return "EN"

    return "OTHER"


def is_real_word(token):
    """
    Keep alphabetic Roman-script tokens.
    Remove punctuation/special tokens.
    """

    token = token.strip()

    if not token:
        return False

    # Roman-script word
    if re.fullmatch(r"[A-Za-z]+(?:['-][A-Za-z]+)*", token):
        return True

    return False


# ============================================================
# TOKEN-LEVEL LID
# ============================================================

all_results = []

total_hi = 0
total_en = 0
total_other = 0
total_words = 0

mixed_sentences = 0
monolingual_sentences = 0

total_switches = 0

print("Running token-level Hindi/English identification...")
print()

for start in range(0, len(texts), BATCH_SIZE):

    batch_texts = texts[start:start + BATCH_SIZE]

    encoded = tokenizer(
        batch_texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
        return_attention_mask=True
    )

    encoded = {
        key: value.to(DEVICE)
        for key, value in encoded.items()
    }

    with torch.no_grad():
        outputs = model(**encoded)

    predictions = torch.argmax(
        outputs.logits,
        dim=-1
    )

    batch_size = len(batch_texts)

    for i in range(batch_size):

        input_ids = encoded["input_ids"][i]
        attention_mask = encoded["attention_mask"][i]
        pred_ids = predictions[i]

        tokens = tokenizer.convert_ids_to_tokens(
            input_ids
        )

        word_labels = []

        for token, pred_id, mask in zip(
            tokens,
            pred_ids.tolist(),
            attention_mask.tolist()
        ):

            if mask == 0:
                continue

            # Ignore BERT special tokens
            if token in tokenizer.all_special_tokens:
                continue

            # Ignore subword pieces
            if token.startswith("##"):
                continue

            if not is_real_word(token):
                continue

            label = clean_label(
                id2label[int(pred_id)]
            )

            word_labels.append(
                (token, label)
            )

        hi_count = sum(
            1 for _, label in word_labels
            if label == "HI"
        )

        en_count = sum(
            1 for _, label in word_labels
            if label == "EN"
        )

        other_count = sum(
            1 for _, label in word_labels
            if label == "OTHER"
        )

        word_count = (
            hi_count +
            en_count +
            other_count
        )

        # ----------------------------------------------------
        # Sentence-level code mixing
        # ----------------------------------------------------

        languages = [
            label
            for _, label in word_labels
            if label in {"HI", "EN"}
        ]

        is_mixed = (
            "HI" in languages
            and
            "EN" in languages
        )

        if is_mixed:
            mixed_sentences += 1
        else:
            monolingual_sentences += 1

        # ----------------------------------------------------
        # Switching points
        # ----------------------------------------------------

        switches = 0

        previous = None

        for _, label in word_labels:

            if label not in {"HI", "EN"}:
                continue

            if previous is not None and label != previous:
                switches += 1

            previous = label

        total_switches += switches

        # ----------------------------------------------------
        # Sentence CMI
        #
        # CMI = 100 * (1 - max(HI, EN) / total_words)
        # calculated over Hindi/English words
        # ----------------------------------------------------

        hi_en_total = hi_count + en_count

        if hi_en_total > 0:

            sentence_cmi = (
                100.0 *
                (
                    1.0 -
                    max(hi_count, en_count)
                    / hi_en_total
                )
            )

        else:
            sentence_cmi = 0.0

        # ----------------------------------------------------
        # Token ratios
        # ----------------------------------------------------

        if hi_en_total > 0:

            hi_ratio = hi_count / hi_en_total
            en_ratio = en_count / hi_en_total

        else:

            hi_ratio = 0.0
            en_ratio = 0.0

        total_hi += hi_count
        total_en += en_count
        total_other += other_count
        total_words += word_count

        all_results.append({
            "text": batch_texts[i],
            "hindi_tokens": hi_count,
            "english_tokens": en_count,
            "other_tokens": other_count,
            "total_tokens": word_count,
            "hindi_ratio": hi_ratio,
            "english_ratio": en_ratio,
            "cmi": sentence_cmi,
            "switches": switches,
            "is_code_mixed": is_mixed
        })

    processed = min(
        start + BATCH_SIZE,
        len(texts)
    )

    if processed % 1000 < BATCH_SIZE or processed == len(texts):

        print(
            f"Processed: {processed:,}/{len(texts):,}"
        )


# ============================================================
# OVERALL METRICS
# ============================================================

hi_en_total = total_hi + total_en

if hi_en_total > 0:

    overall_hindi_ratio = (
        total_hi / hi_en_total
    )

    overall_english_ratio = (
        total_en / hi_en_total
    )

    overall_cmi = (
        100.0 *
        (
            1.0 -
            max(total_hi, total_en)
            / hi_en_total
        )
    )

else:

    overall_hindi_ratio = 0.0
    overall_english_ratio = 0.0
    overall_cmi = 0.0


num_sentences = len(texts)

mixed_percentage = (
    100.0 * mixed_sentences / num_sentences
    if num_sentences
    else 0.0
)

average_switches = (
    total_switches / num_sentences
    if num_sentences
    else 0.0
)

average_cmi = (
    sum(r["cmi"] for r in all_results)
    / num_sentences
    if num_sentences
    else 0.0
)


# ============================================================
# RESULTS
# ============================================================

summary = {
    "model": MODEL_NAME,
    "validation_samples": num_sentences,

    "total_tokens": total_words,

    "hindi_tokens": total_hi,
    "english_tokens": total_en,
    "other_tokens": total_other,

    "overall_hindi_ratio": overall_hindi_ratio,
    "overall_english_ratio": overall_english_ratio,

    "overall_cmi": overall_cmi,
    "average_sentence_cmi": average_cmi,

    "mixed_sentences": mixed_sentences,
    "monolingual_sentences": monolingual_sentences,
    "mixed_sentence_percentage": mixed_percentage,

    "total_switches": total_switches,
    "average_switches_per_sentence": average_switches
}


print()
print("=" * 70)
print("LINGUISTIC EVALUATION RESULTS")
print("=" * 70)

for key, value in summary.items():
    print(f"{key}: {value}")

print("=" * 70)


# ============================================================
# SAVE SENTENCE-LEVEL RESULTS
# ============================================================

results_df = pd.DataFrame(all_results)

csv_path = os.path.join(
    OUTPUT_DIR,
    "roman_hinglish_linguistic_results.csv"
)

results_df.to_csv(
    csv_path,
    index=False,
    encoding="utf-8"
)


# ============================================================
# SAVE SUMMARY
# ============================================================

json_path = os.path.join(
    OUTPUT_DIR,
    "roman_hinglish_linguistic_summary.json"
)

with open(
    json_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=2,
        ensure_ascii=False
    )


print()
print("Saved files:")
print(csv_path)
print(json_path)
print()
print("DONE.")