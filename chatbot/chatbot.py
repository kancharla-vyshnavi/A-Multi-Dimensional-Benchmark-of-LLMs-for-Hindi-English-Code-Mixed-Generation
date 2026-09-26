from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch


# ============================================================
# MODEL PATHS
# ============================================================

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"

ADAPTER = (
    r"C:\Users\vyshu\OneDrive\Desktop\HinglishLLM\data\model"
    r"\phi35_mini_final_17340\content\drive\MyDrive\HindiHinglish_LLM"
    r"\outputs\phi35_mini_final_17340\qwen25_3b_cpt\checkpoint-17340"
)


# ============================================================
# GPU CHECK
# ============================================================

print("===================================")
print("       HinglishLLM Chatbot")
print("===================================")

print("\nChecking GPU...")

if torch.cuda.is_available():
    print("CUDA: Available")
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM:", round(
        torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 1
    ), "GB")
else:
    print("CUDA: Not available")
    print("Using CPU")


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL
)


# ============================================================
# LOAD BASE MODEL
# ============================================================

print("Loading Qwen2.5-3B base model...")

if torch.cuda.is_available():

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float16,
        device_map="auto"
    )

else:

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float32
    )


# ============================================================
# LOAD TRAINED QLoRA ADAPTER
# ============================================================

print("Loading trained Hinglish QLoRA adapter...")

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER
)

model.eval()


# ============================================================
# FIND ACTUAL MODEL DEVICE
# ============================================================

model_device = next(model.parameters()).device

print("Model device:", model_device)

print("\n===================================")
print("       HinglishLLM Chatbot")
print("       Qwen2.5-3B + Trained QLoRA")
print("===================================")
print("Type 'exit' to stop.")
print("")


# ============================================================
# CHAT LOOP
# ============================================================

while True:

    user_input = input("You: ").strip()


    # Exit
    if user_input.lower() == "exit":
        print("Bot: Bye bro! 👋")
        break


    # Empty input
    if not user_input:
        print("Bot: Please enter a question!\n")
        continue


    # ========================================================
    # HINGLISH SYSTEM PROMPT
    # ========================================================

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful Hinglish chatbot. "
                "Understand Hindi-English code-mixed questions. "
                "When the user uses Hinglish, respond naturally "
                "in Hinglish. "
                "Understand common abbreviations from context. "
                "For example, in a coding or placement context, "
                "DSA means Data Structures and Algorithms. "
                "Keep answers clear, concise and useful."
            )
        },
        {
            "role": "user",
            "content": user_input
        }
    ]


    # ========================================================
    # APPLY CHAT TEMPLATE
    # ========================================================

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


    # ========================================================
    # TOKENIZE
    # ========================================================

    inputs = tokenizer(
        text,
        return_tensors="pt"
    )


    # ========================================================
    # MOVE INPUT TO MODEL DEVICE
    # ========================================================

    inputs = {
        key: value.to(model_device)
        for key, value in inputs.items()
    }


    # ========================================================
    # GENERATE
    # ========================================================

    print("Bot is thinking...")

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id
        )


    # ========================================================
    # REMOVE PROMPT TOKENS
    # ========================================================

    generated_tokens = outputs[
        0
    ][
        inputs["input_ids"].shape[1]:
    ]


    # ========================================================
    # DECODE
    # ========================================================

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()


    print("Bot:", response)
    print()