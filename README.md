# Handwritten Mathematical Formula to LaTeX

**Vision-Language Model Fine-Tuning + Streamlit Application**

Converts images of handwritten mathematical formulas into clean LaTeX code using a fine-tuned SmolVLM-256M-Instruct model.

This project was developed as part of the technical task for the **Multimodal Reasoning for STEM** internship at Huawei.

## Results

| Setup                              | CER     | Change vs Zero-shot |
|------------------------------------|---------|---------------------|
| Zero-shot (baseline)               | 0.1846  | —                   |
| One-shot                           | 0.1990  | -7.8%               |
| SFT (linxy/LaTeX_OCR:train only)   | 0.1881  | -1.9%               |
| SFT (linxy/LaTeX_OCR + MathWriting-human) | **0.1859** | **-0.7%**      |

**Metric:** Character Error Rate (CER) — the most suitable for LaTeX generation quality.

## Features
- Zero-shot and One-shot inference
- Supervised Fine-Tuning (SFT) with **LoRA + 4-bit quantization**
- Real-time **Streamlit web application** (upload photo → rendered LaTeX)
- Tested on the official test set (`linxy/LaTeX_OCR:test`, 70 examples)
- Tested on **real photos** of my own handwritten formulas

## Trained Models (Hugging Face)
- [SmolVLM-256M-SFT-linxy](https://huggingface.co/Azaper/SmolVLM-256M-SFT-linxy)
- [SmolVLM-256M-SFT-linxy-deepcopy](https://huggingface.co/Azaper/SmolVLM-256M-SFT-linxy-deepcopy)

## Quick Start

```bash
git clone https://github.com/vys-leon/vlm-handwriting-to-latex.git
cd vlm-handwriting-to-latex
pip install -r requirements.txt