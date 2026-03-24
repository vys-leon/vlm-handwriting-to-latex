import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Загружаем модель
processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
model = AutoModelForImageTextToText.from_pretrained(
    "HuggingFaceTB/SmolVLM-256M-Instruct",
    torch_dtype=torch.bfloat16
).to(DEVICE)
model.eval()

def infer_formula(image_path: str) -> str:
    image = Image.open(image_path).convert("RGB")
    
    prompt = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Write the LaTeX representation for this image."},
                {"type": "image"}
            ]
        }
    ]
    
    inputs = processor(text=processor.apply_chat_template(prompt, add_generation_prompt=True),
                       images=image, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=150)
        answer = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    latex = answer.split("Assistant:")[-1].strip()
    return latex

if __name__ == "__main__":
    latex = infer_formula("screenshots/my_image.jpg")
    print("LaTeX:", latex)