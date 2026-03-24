import torch
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import jiwer
from transformers import AutoProcessor, AutoModelForImageTextToText

# === Загрузка модели один раз ===
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
model = AutoModelForImageTextToText.from_pretrained(
    "HuggingFaceTB/SmolVLM-256M-Instruct", torch_dtype=torch.bfloat16
).to(DEVICE)
model.eval()

def evaluate_cer(test_dataset, mode="zero_shot", train_dataset=None):
    scores = []
    base_text = "Write the LaTeX representation for this image."
    
    # Подготовка промпта
    if mode == "zero_shot":
        prompt = [{"role": "user", "content": [{"type": "text", "text": base_text}, {"type": "image"}]}]
    else:  # one_shot
        example = train_dataset[0]
        prompt = [
            {"role": "user", "content": [{"type": "text", "text": base_text}, {"type": "image"}]},
            {"role": "assistant", "content": [{"type": "text", "text": example["text"].replace(" ", "")}]},
            {"role": "user", "content": [{"type": "text", "text": base_text}, {"type": "image"}]}
        ]
        example_image = example["image"]

    with torch.no_grad():
        for item in tqdm(test_dataset):
            image = item["image"]
            ref = item["text"].replace(" ", "")

            if mode == "zero_shot":
                inputs = processor(text=processor.apply_chat_template(prompt, add_generation_prompt=True),
                                   images=image, return_tensors="pt").to(DEVICE)
            else:
                inputs = processor(text=processor.apply_chat_template(prompt, add_generation_prompt=True),
                                   images=[example_image, image], return_tensors="pt").to(DEVICE)

            gen = model.generate(**inputs, max_new_tokens=150)
            pred = processor.batch_decode(gen, skip_special_tokens=True)[0].split("Assistant:")[-1].strip()

            cer = jiwer.cer(ref, pred)
            scores.append(cer)

    return np.mean(scores)


if __name__ == "__main__":
    test_ds = load_dataset("linxy/LaTeX_OCR", name="human_handwrite", split="test")
    train_ds = load_dataset("linxy/LaTeX_OCR", name="human_handwrite", split="train")
    
    print("Zero-shot CER:", evaluate_cer(test_ds, "zero_shot"))
    print("One-shot CER:", evaluate_cer(test_ds, "one_shot", train_ds))