"""
HinglishLLM Terminal-Only Multilingual Conversational Chatbot

Features:
- Terminal only
- Qwen2.5-3B-Instruct
- Project CPT/QLoRA adapter checkpoint-17340
- Fast generation
- English -> English
- Hindi -> Hindi
- Hinglish -> Hinglish
- Telugu -> Telugu
- Romanized Telugu -> Romanized Telugu
- Keeps recent conversation history
- Detailed error handling
- exit / quit / bye supported
"""

import sys
import io
import re
import traceback
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# ============================================================
# WINDOWS UTF-8
# ============================================================

if sys.platform.startswith("win"):
    try:
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer,
            encoding="utf-8",
            errors="replace"
        )
        sys.stderr = io.TextIOWrapper(
            sys.stderr.buffer,
            encoding="utf-8",
            errors="replace"
        )
    except Exception:
        pass


# ============================================================
# CONFIGURATION
# ============================================================

BASE_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

PROJECT_DIR = Path(__file__).resolve().parent.parent

# Possible adapter locations
CANDIDATE_ADAPTER_PATHS = [
    PROJECT_DIR
    / "models"
    / "qwen25_3b_cpt"
    / "qwen25_3b_cpt"
    / "checkpoint-17340",

    PROJECT_DIR
    / "models"
    / "qwen25_3b_cpt"
    / "checkpoint-17340",

    PROJECT_DIR
    / "data"
    / "model"
    / "qwen25_3b_cpt"
    / "checkpoint-17340",
]


# ============================================================
# SYSTEM PROMPT
# ============================================================

MANDATORY_SYSTEM_PROMPT = (
    "You are a multilingual conversational assistant. "
    "Detect the language and writing style of every user message. "
    "Respond naturally in the same language and writing style. "
    "Do not force Hinglish unless the user is using Hinglish. "
    "Support English, Hindi, Hinglish, Telugu, and mixed-language input. "
    "Keep responses short, relevant, natural, and conversational. "
    "Do not give unnecessarily long answers."
)


# ============================================================
# LIGHTWEIGHT LANGUAGE DETECTION
# ============================================================

TELUGU_WORDS = {
    "ante",
    "enti",
    "naku",
    "naaku",
    "cheppu",
    "cheppandi",
    "ela",
    "unnaru",
    "unnaavu",
    "kavali",
    "kaavali",
    "undhi",
    "undi",
    "chala",
    "chaala",
    "meeru",
    "chesanu",
    "lekapothe",
    "kadha",
    "emi",
    "avunu",
    "ledu",
    "ani",
    "gurinchi",
    "chesuko",
    "cheskovali",
    "evaru",
    "ekkada",
    "epudu",
    "eppudu",
    "bagunnara",
    "baagunnara",
    "bro",
    "andi",
    "kuda",
    "cheyyali",
    "telugu",
    "ardham",
    "ayindhi",
    "avuthundi",
}


HINDI_WORDS = {
    "kya",
    "hai",
    "hain",
    "kaise",
    "kaisa",
    "kaisi",
    "batao",
    "bataiye",
    "mujhe",
    "thoda",
    "thodi",
    "nahi",
    "nahin",
    "karna",
    "karo",
    "hota",
    "hoti",
    "hote",
    "acha",
    "achha",
    "accha",
    "bhai",
    "baare",
    "bare",
    "mein",
    "kuch",
    "samjha",
    "samjhao",
    "kaun",
    "kahan",
    "kab",
    "kyun",
    "kyu",
    "hoga",
    "hogi",
    "kar",
    "raha",
    "rahi",
    "rahe",
    "wala",
    "wali",
    "wale",
    "yeh",
    "ye",
    "woh",
    "wo",
    "apka",
    "aapka",
    "mera",
    "meri",
    "tere",
    "teri",
    "dost",
    "namaste",
    "madad",
}


def detect_language_style(text: str):
    """
    Detect language/writing style using lightweight rules.
    """

    # Telugu script
    if re.search(r"[\u0C00-\u0C7F]", text):
        return (
            "Telugu script",
            "Respond in Telugu using Telugu script."
        )

    # Hindi Devanagari
    if re.search(r"[\u0900-\u097F]", text):
        return (
            "Hindi Devanagari",
            "Respond in Hindi using Devanagari script."
        )

    # Romanized language detection
    tokens = set(
        re.findall(r"\b[a-zA-Z]+\b", text.lower())
    )

    telugu_matches = tokens.intersection(TELUGU_WORDS)
    hindi_matches = tokens.intersection(HINDI_WORDS)

    # Romanized Telugu
    if (
        len(telugu_matches) > 0
        and len(telugu_matches) >= len(hindi_matches)
    ):
        return (
            "Romanized Telugu / Telugu-English",
            "Respond naturally in Romanized Telugu or Telugu-English."
        )

    # Hinglish
    if len(hindi_matches) > 0:
        return (
            "Hinglish (Hindi-English mix)",
            "Respond naturally in Romanized Hinglish."
        )

    # Default
    return (
        "English",
        "Respond naturally in English."
    )


# ============================================================
# ADAPTER PATH
# ============================================================

def resolve_adapter_path():

    for path in CANDIDATE_ADAPTER_PATHS:

        if not path.exists():
            continue

        safetensors_file = path / "adapter_model.safetensors"
        bin_file = path / "adapter_model.bin"
        config_file = path / "adapter_config.json"

        if (
            safetensors_file.is_file()
            or bin_file.is_file()
            or config_file.is_file()
        ):
            return path

    # Recursive fallback search
    search_dir = PROJECT_DIR / "models" / "qwen25_3b_cpt"

    if search_dir.exists():

        for config in search_dir.rglob("adapter_config.json"):
            return config.parent

    return None


# ============================================================
# MODEL LOADING
# ============================================================

def load_chatbot_model():

    adapter_path = resolve_adapter_path()

    if adapter_path is None:
        raise FileNotFoundError(
            "\nCould not locate Qwen2.5-3B CPT/QLoRA adapter.\n"
            f"Expected under:\n"
            f"{PROJECT_DIR / 'models' / 'qwen25_3b_cpt'}\n"
        )

    # CUDA
    if torch.cuda.is_available():

        device = "cuda"
        torch_dtype = torch.float16

    # CPU fallback
    else:

        device = "cpu"
        torch_dtype = torch.float32

    print("=" * 65)
    print("Initializing HinglishLLM Conversational Assistant")
    print("=" * 65)

    print(f"Base Model        : {BASE_MODEL_NAME}")
    print(f"Adapter Checkpoint: {adapter_path}")
    print(f"Device            : {device.upper()}")
    print(f"Dtype             : {torch_dtype}")

    if torch.cuda.is_available():

        try:
            gpu_name = torch.cuda.get_device_name(0)
            print(f"GPU               : {gpu_name}")
        except Exception:
            pass

    print("=" * 65)

    # Tokenizer
    print("Loading tokenizer...", end="", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_NAME
    )

    print(" [Done]")

    # Base model
    print("Loading base model...", end="", flush=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )

    print(" [Done]")

    # Adapter
    print("Loading CPT/QLoRA adapter...", end="", flush=True)

    model = PeftModel.from_pretrained(
        base_model,
        str(adapter_path)
    )

    model.to(device)
    model.eval()

    print(" [Done]")

    # Ensure cache is enabled
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True

    print("=" * 65)
    print("SYSTEM READY")
    print("Type your message and press Enter.")
    print("Type 'exit', 'quit', or 'bye' to stop.")
    print("=" * 65)

    return tokenizer, model, device, adapter_path


# ============================================================
# RESPONSE GENERATION
# ============================================================

def generate_response(
    tokenizer,
    model,
    device,
    history,
    user_input: str
):

    # Detect style
    detected_style, style_instruction = (
        detect_language_style(user_input)
    )

    # System instruction
    system_content = (
        f"{MANDATORY_SYSTEM_PROMPT}\n"
        f"User language/style: {detected_style}\n"
        f"Instruction: {style_instruction}"
    )

    # --------------------------------------------------------
    # KEEP ONLY RECENT HISTORY
    # --------------------------------------------------------
    #
    # This prevents prompts from becoming larger and slower
    # after many conversation turns.
    #
    recent_history = history[-4:]

    messages = [
        {
            "role": "system",
            "content": system_content
        }
    ]

    for turn in recent_history:

        messages.append(
            {
                "role": turn["role"],
                "content": turn["content"]
            }
        )

    messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    # --------------------------------------------------------
    # CHAT TEMPLATE
    # --------------------------------------------------------

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    # Move tensors to GPU / CPU
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    # --------------------------------------------------------
    # FAST GENERATION
    # --------------------------------------------------------

    with torch.inference_mode():

        outputs = model.generate(
            **inputs,

            # Short answers = faster response
            max_new_tokens=48,

            # Greedy decoding = faster than sampling
            do_sample=False,

            # KV cache = faster autoregressive decoding
            use_cache=True,

            # Stop at EOS
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,

            # No beam search
            num_beams=1,
        )

    # --------------------------------------------------------
    # REMOVE PROMPT TOKENS
    # --------------------------------------------------------

    prompt_length = inputs["input_ids"].shape[-1]

    response_tokens = outputs[0][prompt_length:]

    bot_response = tokenizer.decode(
        response_tokens,
        skip_special_tokens=True
    ).strip()

    return bot_response, detected_style


# ============================================================
# AUTOMATED TEST SUITE
# ============================================================

def run_automated_test_suite(
    tokenizer,
    model,
    device
):

    test_cases = [

        (
            "Hi",
            "English"
        ),

        (
            "What is Python?",
            "English"
        ),

        (
            "AI kya hai bro?",
            "Hinglish (Hindi-English mix)"
        ),

        (
            "Mujhe machine learning ke baare mein batao",
            "Hinglish (Hindi-English mix)"
        ),

        (
            "python ante enti?",
            "Romanized Telugu / Telugu-English"
        ),

        (
            "naku Python gurinchi cheppu",
            "Romanized Telugu / Telugu-English"
        ),

        (
            "आप कैसे हैं?",
            "Hindi Devanagari"
        ),

        (
            "మీరు ఎలా ఉన్నారు?",
            "Telugu script"
        ),
    ]

    print("\n" + "=" * 65)
    print("RUNNING CHATBOT VALIDATION")
    print("=" * 65)

    test_history = []

    for index, (prompt, expected_style) in enumerate(
        test_cases,
        1
    ):

        print(
            f"\n[Test {index}/{len(test_cases)}]"
        )

        print(f"Input         : {prompt}")
        print(f"Expected Style: {expected_style}")

        try:

            response, detected = generate_response(
                tokenizer,
                model,
                device,
                test_history,
                prompt
            )

            print(f"Detected Style: {detected}")
            print(f"Bot Response  : {response}")

            test_history.append(
                {
                    "role": "user",
                    "content": prompt
                }
            )

            test_history.append(
                {
                    "role": "assistant",
                    "content": response
                }
            )

        except Exception as error:

            print("\n================ ERROR ================")
            print(
                f"Error type   : {type(error).__name__}"
            )
            print(
                f"Error message: {error}"
            )

            traceback.print_exc()

            print("========================================")

    print("\n" + "=" * 65)
    print("VALIDATION COMPLETED")
    print("=" * 65)


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    try:

        tokenizer, model, device, adapter_path = (
            load_chatbot_model()
        )

    except Exception as error:

        print("\n================ ERROR ================")

        print(
            f"Error type   : {type(error).__name__}"
        )

        print(
            f"Error message: {error}"
        )

        traceback.print_exc()

        print("========================================")

        sys.exit(1)

    # --------------------------------------------------------
    # TEST MODE
    # --------------------------------------------------------

    if "--test" in sys.argv:

        run_automated_test_suite(
            tokenizer,
            model,
            device
        )

        return

    # --------------------------------------------------------
    # INTERACTIVE CHAT
    # --------------------------------------------------------

    history = []

    while True:

        try:

            user_input = input("You: ").strip()

            if not user_input:
                continue

            # Exit
            if user_input.lower() in {
                "exit",
                "quit",
                "bye"
            }:

                print(
                    "Bot: Goodbye! Have a great day ahead."
                )

                break

            # Generate
            bot_response, detected_style = (
                generate_response(
                    tokenizer,
                    model,
                    device,
                    history,
                    user_input
                )
            )

            print(
                f"Bot: {bot_response}\n"
            )

            # Save history
            history.append(
                {
                    "role": "user",
                    "content": user_input
                }
            )

            history.append(
                {
                    "role": "assistant",
                    "content": bot_response
                }
            )

        except (
            KeyboardInterrupt,
            EOFError
        ):

            print(
                "\nBot: Session ended. Goodbye!"
            )

            break

        except Exception as error:

            print(
                "\n================ ERROR ================"
            )

            print(
                f"Error type   : {type(error).__name__}"
            )

            print(
                f"Error message: {error}"
            )

            traceback.print_exc()

            print(
                "========================================"
            )

            print(
                "You can continue chatting or type 'exit'.\n"
            )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
