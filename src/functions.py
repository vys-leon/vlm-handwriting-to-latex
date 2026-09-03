import numpy as np
import pandas as pd
import torch
import jiwer
import re
import random
from transformers import set_seed
from tqdm.notebook import tqdm

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)

def prompt_generation(
    mode="_0shot",
    examples=[]
):
    base_prompt = """
    Convert the handwritten mathematical formula in the image into LaTeX.
    """

    prompt = [
        {
            "role": "user",
            "content" : [

            ]
        }
    ]
    if mode == "_0shot":
      examples=[]
      prompt = [
          { "role": "user",
           "content" : [
               {"type" : "image"},
               {"type" : "text",  "text" : base_prompt}
               ]
            }
          ]
    elif mode == "_1shot":
      example_item = examples[0]
      example_image = example_item['image']
      example_text = example_item['text']

      prompt = [
        {
          "role": "user",
          "content": [
              {"type": "image"},
              {"type": "text", "text": base_prompt}
          ]
        },
        {
          "role": "assistant",
          "content": [
              {"type": "text", "text": example_text}
          ]
        },
        {
          "role": "user",
          "content": [
              {"type": "image"},
              {"type": "text", "text": base_prompt}
          ]
        }
      ]
      examples = [example_image]

    return [prompt, examples]

def generate_prediction(
    model,
    processor,
    images,
    prompt,
    device,
    max_new_tokens=150,
):
    inputs = processor(
        text=prompt,
        images=images,
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )

    prediction = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0]

    return prediction

def normalize_prediction(text: str) -> str:
    text = text.strip()

    # Remove user/assistance construction
    text = re.sub(r"^(?:user)[\s*\S*\s*]*(?:assistant)\s*", "", text)

    # Remove Markdown code fences
    text = re.sub(r"```(?:latex|tex)?\s*", "", text)
    text = re.sub(r"\s*```", "", text)

    # Remove chat special tokens
    text = text.replace("<|im_end|>", "")
    text = text.replace("<|endoftext|>", "")

    return text.strip()

def normalize_reference(text: str) -> str:
    return text.strip()

def evaluate_model(
    model,
    processor,
    dataset,
    device,
    prompt,
    examples=None,
    max_new_tokens=150,
):
    results = []
    prompt, examples = prompt
    print("Evaluating model on the given dataset")
    for idx, item in enumerate(tqdm(dataset)):
        raw_prediction = generate_prediction(
            model=model,
            processor=processor,
            images=examples + [item["image"]],
            prompt=prompt,
            device=device,
            max_new_tokens=max_new_tokens,
        )
        prediction = normalize_prediction(raw_prediction)
        reference = normalize_reference(item["text"])

        results.append({
            "index": idx,
            "prediction": prediction,
            "reference": reference,
            "cer": jiwer.cer(reference, prediction),
        })

    return pd.DataFrame(results)