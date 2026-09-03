import streamlit as st
from PIL import Image
from unsloth import FastVisionModel
from src.functions import *
import os
from dotenv import load_dotenv

@st.cache_resource
def load_model():
    load_dotenv()
    assert "HF_TOKEN" not in os.environ, "Ошибка: Токен HF_TOKEN не найден в файле .env"
    model_id = "Azaper/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit-linxy"
    model, processor = FastVisionModel.from_pretrained(
        model_id,
        load_in_4bit = True,
        use_gradient_checkpointing = "unsloth",
        device_map="auto"
    )
    return processor, model

st.set_page_config(page_title="Handwritten → LaTeX", layout="centered")
st.title("Handwritten Formula to LaTeX")
st.caption("Fine-tuned Qwen3-VL-2B-Instruct")
uploaded_file = st.file_uploader("Upload a photo of the handwritten formula", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Your formula", width="stretch")
    if st.button("Convert to LaTeX", type="primary"):
        with st.spinner("The model is working..."):
            processor, model = load_model()
            prompt = prompt_generation()
            prompt, examples = prompt
            raw_prediction = generate_prediction(
                model=model,
                processor=processor,
                images=examples + [image],
                prompt=prompt,
                device=model.device,
                max_new_tokens=2048,
                )
            prediction = normalize_prediction(raw_prediction)
            
            st.success("Ready!")
            st.latex(prediction)
            st.code(prediction, language="latex")
            st.download_button("Download LaTeX", prediction, file_name="formula.tex")