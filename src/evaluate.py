from datasets import load_dataset
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

if __name__ == "__main__":
    test_dataset_linxy = load_dataset("linxy/LaTeX_OCR", name="human_handwrite", split="test")
    prompt = prompt_generation()
    results = evaluate_model(model=model, processor=tokenizer, dataset=test_dataset_linxy, device=model.device, prompt=prompt)
    cer = results["cer"].mean()
    print("CER on test linxy dataset: ", cer)