import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

st.set_page_config(
    page_title="Transformer Translator",
    page_icon="🌍",
    layout="centered"
)

st.title("🌍 Transformer-Based Text Translation")
st.write("Translate English text into multiple languages using Transformers.")

MODELS = {
    "Hindi": "Helsinki-NLP/opus-mt-en-hi",
    "French": "Helsinki-NLP/opus-mt-en-fr",
    "German": "Helsinki-NLP/opus-mt-en-de",
    "Spanish": "Helsinki-NLP/opus-mt-en-es",
    "Italian": "Helsinki-NLP/opus-mt-en-it"
}

@st.cache_resource
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

target_language = st.selectbox(
    "Select Target Language",
    list(MODELS.keys())
)

model_name = MODELS[target_language]

with st.spinner("Loading model..."):
    tokenizer, model = load_model(model_name)

input_text = st.text_area(
    "Enter English Text",
    height=150,
    placeholder="Type your text here..."
)

if st.button("Translate"):

    if not input_text.strip():
        st.warning("Please enter some text.")

    else:
        with st.spinner("Translating..."):

            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )

            translated_tokens = model.generate(
                **inputs,
                max_length=512,
                num_beams=4,
                early_stopping=True
            )

            translated_text = tokenizer.decode(
                translated_tokens[0],
                skip_special_tokens=True
            )

        st.success("Translation Completed!")

        st.subheader("Translated Text")
        st.write(translated_text)

st.markdown("---")
st.caption("Built using Hugging Face Transformers and Streamlit")
