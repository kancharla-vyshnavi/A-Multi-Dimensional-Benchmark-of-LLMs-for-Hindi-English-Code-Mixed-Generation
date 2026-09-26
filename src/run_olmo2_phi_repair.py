import csv
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.stdout.reconfigure(encoding="utf-8")

# ============================================================
# PATHS AND CONFIG
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent
INPUT_CSV = ROOT_DIR / "outputs" / "controlled_1000" / "benchmark_generations_.csv"
OUTPUT_DIR = ROOT_DIR / "outputs" / "independent_judge_1000"
REPAIRED_PHI_CSV = OUTPUT_DIR / "independent_olmo2_1b_judge__phi_repaired.csv"
CORRECTED_FINAL_CSV = OUTPUT_DIR / "independent_olmo2_1b_judge__CORRECTED_FINAL.csv"
EXISTING_FINAL_CSV = OUTPUT_DIR / "independent_olmo2_1b_judge__FINAL.csv"

JUDGE_MODEL = "allenai/OLMo-2-0425-1B-Instruct"
SCORE_FIELDS = [
    "fluency",
    "code_mixing_naturalness",
    "hindi_grammar",
    "prompt_adherence",
    "spelling_consistency",
    "overall",
]
BATCH_SIZE = 4
MAX_INPUT_TOKENS = 512
MAX_NEW_TOKENS = 96
SEED = 42

torch.manual_seed(SEED)

# ============================================================
# PROMPT BUILDER
# ============================================================

def build_judge_prompt(prompt, response):
    return f"""
You are an independent evaluator of Hindi-English code-mixed text generation.

Evaluate ONLY the model response against the user prompt.

USER PROMPT:
{prompt}

MODEL RESPONSE:
{response}

Score each dimension from 1 to 5.

1. fluency
2. code_mixing_naturalness
3. hindi_grammar
4. prompt_adherence
5. spelling_consistency
6. overall

Scoring:
1 = very poor
2 = poor
3 = acceptable
4 = good
5 = excellent

Focus on the quality of the response itself.
Do not compare models.
Do not reward or penalize the response merely for being short or long.
Consider natural Hindi-English code-mixing appropriate to the prompt.

Return ONLY one valid JSON object.
No explanation.
No markdown.
Do not copy any example.
Values for all six fields MUST be integers from 1 to 5.
Evaluate the actual response before assigning scores.

Required field names:
fluency
code_mixing_naturalness
hindi_grammar
prompt_adherence
spelling_consistency
overall
""".strip()

def parse_scores(text):
    text = str(text).strip()
    text = re.sub(r"```json", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```", "", text).strip()

    candidates = []
    try:
        candidates.append(json.loads(text))
    except Exception:
        pass

    # Find any JSON-like substrings
    for match in re.finditer(r"\{[^{}]*\}", text):
        try:
            candidates.append(json.loads(match.group(0)))
        except Exception:
            pass

    for obj in candidates:
        if not isinstance(obj, dict):
            continue
        result = {}
        valid = True
        for field in SCORE_FIELDS:
            value = obj.get(field)
            try:
                value = int(float(value))
            except Exception:
                valid = False
                break
            if value < 1 or value > 5:
                valid = False
                break
            result[field] = value
        if valid:
            return result
    return None

def main():
    print("=" * 80)
    print("OLMo-2-0425-1B-Instruct PHI-3.5-MINI EVALUATION REPAIR")
    print("=" * 80)
    print("Judge Model   :", JUDGE_MODEL)
    print("Input CSV     :", INPUT_CSV)
    print("Repaired CSV  :", REPAIRED_PHI_CSV)
    print("Corrected CSV :", CORRECTED_FINAL_CSV)
    print("CUDA Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Device        :", torch.cuda.get_device_name(0))
    print("=" * 80)

    # 1. Load input generations
    df = pd.read_csv(INPUT_CSV)
    phi_df = df[df["model"] == "Phi-3.5-mini"].sort_values("row_index").reset_index(drop=True)

    print(f"\nTotal Phi-3.5-mini input rows: {len(phi_df)}")
    assert len(phi_df) == 1000, f"Expected 1000 Phi rows, got {len(phi_df)}"
    assert not phi_df["response"].astype(str).str.contains("GENERATION_ERROR").any(), "Found GENERATION_ERROR in input!"

    # 2. Check existing progress if resume needed
    completed_rows = {}
    if REPAIRED_PHI_CSV.exists():
        try:
            existing = pd.read_csv(REPAIRED_PHI_CSV)
            if len(existing) > 0 and "overall" in existing.columns:
                for _, r in existing.iterrows():
                    if pd.notna(r["overall"]):
                        completed_rows[int(r["row_index"])] = r.to_dict()
                print(f"Resuming from existing progress: {len(completed_rows)} already completed.")
        except Exception as e:
            print(f"Could not load existing progress: {e}")

    # 3. Load Model and Tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        JUDGE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    print("Model loaded successfully.")

    # 4. Run inference on pending rows
    pending_indices = [idx for idx in range(len(phi_df)) if int(phi_df.iloc[idx]["row_index"]) not in completed_rows]
    print(f"Pending rows to evaluate: {len(pending_indices)}")

    start_time = time.time()
    results_map = completed_rows.copy()

    for b_start in range(0, len(pending_indices), BATCH_SIZE):
        batch_slice = pending_indices[b_start : b_start + BATCH_SIZE]
        batch_rows = [phi_df.iloc[i] for i in batch_slice]

        prompts = [
            build_judge_prompt(str(r["prompt"]), str(r["response"]))
            for r in batch_rows
        ]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_INPUT_TOKENS
        ).to("cuda")

        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        decoded = tokenizer.batch_decode(
            generated[:, input_len:],
            skip_special_tokens=True
        )

        for r, raw_output in zip(batch_rows, decoded):
            scores = parse_scores(raw_output)
            row_idx = int(r["row_index"])

            res_dict = {
                "row_index": row_idx,
                "model": "Phi-3.5-mini",
                "prompt": str(r["prompt"]),
                "response": str(r["response"]),
                "judge_model": JUDGE_MODEL,
                "judge_prompt_attempt": None,
                "raw_judge_output": raw_output.strip(),
            }

            for f in SCORE_FIELDS:
                res_dict[f] = scores[f] if scores is not None else None

            results_map[row_idx] = res_dict

        # Periodic save and print
        done_count = len(results_map)
        if (b_start // BATCH_SIZE) % 10 == 0 or done_count == 1000:
            elapsed = time.time() - start_time
            rate = (b_start + len(batch_slice)) / max(elapsed, 1e-5)
            print(f"Progress: [{done_count}/1000] ({rate:.1f} rows/s) - row {batch_rows[-1]['row_index']} overall: {results_map[int(batch_rows[-1]['row_index'])]['overall']}")

            # Save repaired Phi intermediate
            out_df = pd.DataFrame(list(results_map.values())).sort_values("row_index")
            out_df.to_csv(REPAIRED_PHI_CSV, index=False, encoding="utf-8-sig")

    # Final save of repaired Phi
    phi_repaired_df = pd.DataFrame(list(results_map.values())).sort_values("row_index").reset_index(drop=True)
    # Ensure correct column ordering matching FINAL.csv
    cols = [
        "row_index", "model", "prompt", "response",
        "fluency", "code_mixing_naturalness", "hindi_grammar",
        "prompt_adherence", "spelling_consistency", "overall",
        "judge_model", "judge_prompt_attempt", "raw_judge_output"
    ]
    phi_repaired_df = phi_repaired_df[cols]
    phi_repaired_df.to_csv(REPAIRED_PHI_CSV, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print(f"REPAIRED PHI SAVED TO: {REPAIRED_PHI_CSV}")
    print(f"Total rows: {len(phi_repaired_df)}")
    print(f"Valid overall scores: {phi_repaired_df['overall'].notna().sum()}/1000")
    print(f"Mean overall score  : {phi_repaired_df['overall'].mean():.3f}")
    print("=" * 80)

    # 5. Merge with existing 3 models into CORRECTED_FINAL_CSV
    print("\nMerging repaired Phi with existing valid models...")
    existing_final_df = pd.read_csv(EXISTING_FINAL_CSV)
    print(f"Loaded existing FINAL.csv with {len(existing_final_df)} rows.")

    unaffected_df = existing_final_df[existing_final_df["model"] != "Phi-3.5-mini"].copy()
    print("Unaffected model counts in existing FINAL:")
    print(unaffected_df["model"].value_counts())
    assert len(unaffected_df) == 3000, f"Expected 3000 rows for 3 models, got {len(unaffected_df)}"

    # Combine unaffected + repaired Phi
    corrected_final_df = pd.concat([unaffected_df, phi_repaired_df], ignore_index=True)
    # Sort by row_index, model
    corrected_final_df = corrected_final_df.sort_values(by=["row_index", "model"]).reset_index(drop=True)

    corrected_final_df.to_csv(CORRECTED_FINAL_CSV, index=False, encoding="utf-8-sig")
    print(f"\nCORRECTED FINAL SAVED TO: {CORRECTED_FINAL_CSV}")
    print(f"Total rows: {len(corrected_final_df)}")
    print("Model counts in CORRECTED FINAL:")
    print(corrected_final_df["model"].value_counts())
    print("\nOverall score means by model:")
    print(corrected_final_df.groupby("model")["overall"].mean())
    print("\nGeneration error count in CORRECTED FINAL:")
    gen_errors = corrected_final_df["response"].astype(str).str.contains("GENERATION_ERROR").sum()
    print(f"GENERATION_ERROR count: {gen_errors}")
    assert gen_errors == 0, "Found GENERATION_ERROR in CORRECTED FINAL!"
    print("ALL INTEGRITY CHECKS PASSED!")

if __name__ == "__main__":
    main()
