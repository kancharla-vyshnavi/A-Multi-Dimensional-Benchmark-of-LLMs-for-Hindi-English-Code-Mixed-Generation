# HinglishLLM Terminal Multilingual Chatbot (final)
#
# - Qwen2.5-3B-Instruct + CPT/QLoRA adapter (checkpoint-17340), merged for speed
# - English / Hindi / Hinglish / Telugu / Romanized Telugu -> same style reply
# - Streaming output, short history, error handling
# - Run:  python chatbot\app.py          (chat)
#         python chatbot\app.py --test   (automated test)

import sys
import re
import traceback
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextStreamer,
)
from peft import PeftModel


# ---------------- Windows UTF-8 ----------------
for _s in (sys.stdout, sys.stderr, sys.stdin):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# ---------------- CONFIG ----------------
BASE_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
PROJECT_DIR = Path(__file__).resolve().parent.parent

CANDIDATE_ADAPTER_PATHS = [
    PROJECT_DIR / "models" / "qwen25_3b_cpt" / "qwen25_3b_cpt" / "checkpoint-17340",
    PROJECT_DIR / "models" / "qwen25_3b_cpt" / "checkpoint-17340",
    PROJECT_DIR / "data" / "model" / "qwen25_3b_cpt" / "checkpoint-17340",
]

HISTORY_MESSAGES = 6          # last 3 turns
MAX_NEW_TOKENS_LATIN = 160
MAX_NEW_TOKENS_SCRIPT = 256   # Telugu/Hindi script uses many more tokens

SYSTEM_PROMPT = (
    "You are a multilingual conversational assistant. "
    "Detect the language and writing style of every user message. "
    "Respond naturally in the same language and writing style. "
    "Do not force Hinglish unless the user is using Hinglish. "
    "Support English, Hindi, Hinglish, Telugu, and mixed-language input. "
    "Keep responses short, relevant, natural, and conversational."
)


# ---------------- LANGUAGE DETECTION ----------------
TELUGU_WORDS = {
    "ante", "enti", "naku", "naaku", "cheppu", "cheppandi", "ela", "unnaru",
    "unnaavu", "kavali", "kaavali", "undhi", "undi", "chala", "chaala",
    "meeru", "chesanu", "lekapothe", "kadha", "emi", "avunu", "ledu",
    "gurinchi", "chesuko", "cheskovali", "evaru", "ekkada", "epudu",
    "eppudu", "bagunnara", "baagunnara", "andi", "kuda", "cheyyali",
    "ardham", "ayindhi", "avuthundi", "nenu", "nuvvu", "meku", "cheyyi",
}

HINDI_WORDS = {
    "kya", "hai", "hain", "kaise", "kaisa", "kaisi", "batao", "bataiye",
    "mujhe", "thoda", "thodi", "nahi", "nahin", "karna", "karo", "hota",
    "hoti", "hote", "acha", "achha", "accha", "bhai", "baare", "mein",
    "kuch", "samjha", "samjhao", "kaun", "kahan", "kab", "kyun", "kyu",
    "hoga", "hogi", "raha", "rahi", "rahe", "wali", "wale", "yeh", "woh",
    "apka", "aapka", "mera", "meri", "tere", "teri", "dost", "namaste",
    "madad", "aap", "tum", "main", "ke", "ka", "ki",
}


def detect_language_style(text: str):
    # Returns (style_name, instruction, is_script)
    if re.search(r"[\u0C00-\u0C7F]", text):
        return "Telugu script", "Respond in Telugu using Telugu script.", True

    if re.search(r"[\u0900-\u097F]", text):
        return "Hindi Devanagari", "Respond in Hindi using Devanagari script.", True

    tokens = set(re.findall(r"[a-zA-Z]+", text.lower()))
    te = len(tokens & TELUGU_WORDS)
    hi = len(tokens & HINDI_WORDS)

    if te > 0 and te > hi:
        return (
            "Romanized Telugu / Telugu-English",
            "Respond naturally in Romanized Telugu (Telugu written in English letters).",
            False,
        )
    if hi > 0:
        return (
            "Hinglish (Hindi-English mix)",
            "Respond naturally in Romanized Hinglish.",
            False,
        )
    return "English", "Respond naturally in English.", False


# ---------------- ADAPTER PATH ----------------
def resolve_adapter_path():
    for path in CANDIDATE_ADAPTER_PATHS:
        if path.exists() and (path / "adapter_config.json").is_file():
            return path

    search_dir = PROJECT_DIR / "models" / "qwen25_3b_cpt"
    if search_dir.exists():
        for config in search_dir.rglob("adapter_config.json"):
            return config.parent
    return None


# ---------------- MODEL LOADING ----------------
def load_chatbot_model():
    adapter_path = resolve_adapter_path()
    if adapter_path is None:
        raise FileNotFoundError(
            "Could not find the Qwen2.5-3B CPT/QLoRA adapter under:\n"
            f"{PROJECT_DIR / 'models' / 'qwen25_3b_cpt'}"
        )

    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
        torch.backends.cuda.matmul.allow_tf32 = True
    else:
        device = "cpu"
        dtype = torch.float32

    print("=" * 65)
    print("Initializing HinglishLLM Conversational Assistant")
    print("=" * 65)
    print(f"Base Model : {BASE_MODEL_NAME}")
    print(f"Adapter    : {adapter_path}")
    print(f"Device     : {device.upper()}  |  Dtype: {dtype}")
    if device == "cuda":
        print(f"GPU        : {torch.cuda.get_device_name(0)}")
    print("=" * 65)

    print("Loading tokenizer...", end="", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    print(" [Done]")

    print("Loading base model...", end="", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    print(" [Done]")

    print("Loading + merging adapter...", end="", flush=True)
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model = model.merge_and_unload()   # merge LoRA into base = faster inference
    model.eval()
    model.config.use_cache = True
    print(" [Done]")

    # Warm-up so the first real reply isn't slow
    try:
        warm = tokenizer("Hi", return_tensors="pt").to(device)
        with torch.inference_mode():
            model.generate(**warm, max_new_tokens=2, do_sample=False)
    except Exception:
        pass

    print("=" * 65)
    print("SYSTEM READY  (type 'exit', 'quit' or 'bye' to stop)")
    print("=" * 65)
    return tokenizer, model, device


# ---------------- GENERATION ----------------
def generate_response(tokenizer, model, device, history, user_input, stream=False):
    style, instruction, is_script = detect_language_style(user_input)

    messages = [
        {
            "role": "system",
            "content": (
                f"{SYSTEM_PROMPT}\n"
                f"User language/style: {style}\n"
                f"Instruction: {instruction}"
            ),
        }
    ]
    messages.extend(history[-HISTORY_MESSAGES:])
    messages.append({"role": "user", "content": user_input})

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    streamer = (
        TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        if stream
        else None
    )

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS_SCRIPT if is_script else MAX_NEW_TOKENS_LATIN,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            repetition_penalty=1.08,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            streamer=streamer,
        )

    prompt_len = inputs["input_ids"].shape[-1]
    reply = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True).strip()
    return reply, style


# ---------------- TEST SUITE ----------------
def run_automated_test_suite(tokenizer, model, device):
    tests = [
        ("Hi", "English"),
        ("What is Python?", "English"),
        ("AI kya hai bro?", "Hinglish (Hindi-English mix)"),
        ("Mujhe machine learning ke baare mein batao", "Hinglish (Hindi-English mix)"),
        ("python ante enti?", "Romanized Telugu / Telugu-English"),
        ("naku Python gurinchi cheppu", "Romanized Telugu / Telugu-English"),
        ("आप कैसे हैं?", "Hindi Devanagari"),
        ("మీరు ఎలా ఉన్నారు?", "Telugu script"),
    ]

    print("\n" + "=" * 65)
    print("RUNNING CHATBOT VALIDATION")
    print("=" * 65)

    history, passed = [], 0
    for i, (prompt, expected) in enumerate(tests, 1):
        print(f"\n[Test {i}/{len(tests)}]")
        print(f"Input         : {prompt}")
        print(f"Expected Style: {expected}")
        try:
            reply, detected = generate_response(
                tokenizer, model, device, history, prompt
            )
            ok = detected == expected
            passed += ok
            print(f"Detected Style: {detected}  {'OK' if ok else 'MISMATCH'}")
            print(f"Bot Response  : {reply}")
            if reply:
                history += [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": reply},
                ]
        except Exception as e:
            print(f"ERROR {type(e).__name__}: {e}")
            traceback.print_exc()

    print("\n" + "=" * 65)
    print(f"VALIDATION COMPLETED  |  Language detection: {passed}/{len(tests)}")
    print("=" * 65)


# ---------------- MAIN ----------------
def main():
    try:
        tokenizer, model, device = load_chatbot_model()
    except Exception as e:
        print(f"\nERROR {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)

    if "--test" in sys.argv:
        run_automated_test_suite(tokenizer, model, device)
        return

    history = []
    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue

            if user_input.lower() in {"exit", "quit", "bye"}:
                print("Bot: Goodbye! Have a great day ahead.")
                break

            print("Bot: ", end="", flush=True)
            reply, _ = generate_response(
                tokenizer, model, device, history, user_input, stream=True
            )
            print()

            if reply:
                history += [
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": reply},
                ]

        except (KeyboardInterrupt, EOFError):
            print("\nBot: Session ended. Goodbye!")
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            history.clear()
            print("\nGPU out of memory. History cleared, try again.\n")
        except Exception as e:
            print(f"\nERROR {type(e).__name__}: {e}")
            traceback.print_exc()
            print("You can continue chatting or type 'exit'.\n")


if __name__ == "__main__":
    main()