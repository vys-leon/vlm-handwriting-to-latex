import streamlit as st
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

@st.cache_resource
def load_model():
    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
    model = AutoModelForImageTextToText.from_pretrained(
        "HuggingFaceTB/SmolVLM-256M-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    return processor, model

st.set_page_config(page_title="Handwritten → LaTeX", layout="centered")

st.title("Handwritten Formula to LaTeX")
st.caption("Fine-tuned SmolVLM-256M")

uploaded_file = st.file_uploader("Загрузите фото рукописной формулы", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Ваша формула", use_column_width=True)
    
    if st.button("Преобразовать в LaTeX", type="primary"):
        with st.spinner("Модель работает..."):
            processor, model = load_model()
            
            prompt = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Convert this handwritten formula to LaTeX. Output ONLY the LaTeX code, nothing else."}
                ]}
            ]
            
            inputs = processor(text=processor.apply_chat_template(prompt, add_generation_prompt=True),
                               images=image, return_tensors="pt").to(model.device)
            
            generated_ids = model.generate(**inputs, max_new_tokens=200)
            answer = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            latex = answer.split("Assistant:")[-1].strip()
            
            st.success("Готово!")
            st.latex(latex)
            st.code(latex, language="latex")
            
            st.download_button("Скачать LaTeX", latex, file_name="formula.tex")