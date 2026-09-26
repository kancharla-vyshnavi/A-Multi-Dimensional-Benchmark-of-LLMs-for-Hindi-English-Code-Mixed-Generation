import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

st.set_page_config(page_title="HinglishLLM Chatbot", page_icon="??")

st.title("?? HinglishLLM Chatbot")
st.caption("Powered by Qwen2.5-7B-Instruct")

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return tokenizer, model

tokenizer, model = load_model()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": "You are a helpful Hinglish chatbot. Respond naturally using Hindi-English code-mixed language when appropriate."
        }
    ]

for message in st.session_state.messages[1:]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask something in Hinglish...")

if prompt:
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Generating..."):
            inputs = tokenizer.apply_chat_template(
                st.session_state.messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True
                )

            response = tokenizer.decode(
                outputs[0][inputs.shape[-1]:],
                skip_special_tokens=True
            )

            st.markdown(response)

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )
