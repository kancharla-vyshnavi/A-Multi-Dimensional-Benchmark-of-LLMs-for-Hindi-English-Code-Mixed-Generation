import pandas as pd
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

INPUT = r".\outputs\llm_judge\hinglish_bench_llm_judge_results_final.csv"
OUTPUT = r".\outputs\llm_judge\hinglish_bench_llm_judge_results_final.csv"

cols = [
    "fluency",
    "code_mixing_naturalness",
    "hindi_grammar",
    "prompt_adherence",
    "spelling_consistency",
    "overall"
]

df = pd.read_csv(INPUT)

targets = [
    ("HingGPT", 3),
    ("HingGPT", 8)
]

JUDGE_MODEL = "microsoft/Phi-3.5-mini-instruct"

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

model = AutoModelForCausalLM.from_pretrained(
    JUDGE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.float16,
    trust_remote_code=True,
    attn_implementation="eager"
)

model.eval()
model.config.use_cache = False

# Exact token IDs for digits 1-5
digit_ids = []
for n in range(1, 6):
    ids = tokenizer.encode(str(n), add_special_tokens=False)
    digit_ids.append(ids[-1])

digit_ids = set(digit_ids)

print("Allowed score token IDs:", digit_ids)

for model_name, sample_id in targets:

    mask = (df["model"] == model_name) & (df["sample_id"] == sample_id)
    idx = df.index[mask][0]
    row = df.loc[idx]

    prompt = f"""
You are a strict evaluator.

Evaluate the following Hinglish response on six dimensions.
Each dimension must receive an integer score from 1 to 5.

Dimensions, in order:
1. Fluency
2. Code-mixing naturalness
3. Hindi grammar
4. Prompt adherence
5. Spelling consistency
6. Overall quality

IMPORTANT:
Output EXACTLY six digits.
Each digit must be one of: 1 2 3 4 5.
Do not output words.
Do not output emojis.
Do not explain anything.

Example:
2 1 1 1 1 1

PROMPT:
{row['prompt']}

RESPONSE:
{row['response']}

SIX SCORES:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=768
    )

    inputs = {k: v.to("cuda:0") for k, v in inputs.items()}

    generated_ids = []

    # Generate one score at a time.
    for position in range(6):

        with torch.no_grad():
            out = model(
                **inputs,
                use_cache=False
            )

        logits = out.logits[:, -1, :]

        # Only allow 1-5 score tokens.
        allowed = torch.full_like(logits, float("-inf"))

        for token_id in digit_ids:
            allowed[:, token_id] = logits[:, token_id]

        next_token = torch.argmax(allowed, dim=-1, keepdim=True)

        generated_ids.append(next_token.item())

        inputs["input_ids"] = torch.cat(
            [inputs["input_ids"], next_token],
            dim=1
        )

        if "attention_mask" in inputs:
            new_mask = torch.ones(
                (inputs["attention_mask"].shape[0], 1),
                dtype=inputs["attention_mask"].dtype,
                device=inputs["attention_mask"].device
            )
            inputs["attention_mask"] = torch.cat(
                [inputs["attention_mask"], new_mask],
                dim=1
            )

    generated_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True
    )

    scores = [int(x) for x in re.findall(r"[1-5]", generated_text)]

    print()
    print(f"{model_name} sample {sample_id}")
    print("RAW:", repr(generated_text))
    print("SCORES:", scores)

    if len(scores) == 6:

        for c, score in zip(cols, scores):
            df.loc[idx, c] = score

        # Keep raw judge output for traceability.
        df.loc[idx, "judge_raw_output"] = generated_text

        print("RECOVERED:", scores)

    else:
        print("FAILED")

    df.to_csv(OUTPUT, index=False)

print()
print("Saved:", OUTPUT)