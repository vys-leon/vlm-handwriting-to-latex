from PIL import Image
from unsloth import FastVisionModel
from .functions import *
import os
from dotenv import load_dotenv

load_dotenv()
assert "HF_TOKEN" not in os.environ, "Ошибка: Токен HF_TOKEN не найден в файле .env"
model_id = "Azaper/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit-linxy"
model, tokenizer = FastVisionModel.from_pretrained(
    model_id,
    load_in_4bit = True,
    use_gradient_checkpointing = "unsloth",
    device_map="auto"
)

def infer_formula(image_path: str) -> str:
    image = Image.open(image_path).convert("RGB")
    prompt = prompt_generation()
    prompt, examples = prompt
    raw_prediction = generate_prediction(
            model=model,
            processor=tokenizer,
            images=examples + [image],
            prompt=prompt,
            device=model.device,
            max_new_tokens=2048,
        )
    prediction = normalize_prediction(raw_prediction)
    return prediction

if __name__ == "__main__":
    image_path = "screenshots/my_image.jpg"
    latex = infer_formula(image_path)
    print(f"LaTeX code for formula at image at the path {image_path}:\n", latex)