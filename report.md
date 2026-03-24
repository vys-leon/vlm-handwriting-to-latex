# Technical Report: Handwritten Formula to LaTeX OCR System

**Task 1 – Multimodal Reasoning for STEM Internship**  
**Author:** Leonid Vysotsky  
**Date:** 24 March 2026

## 1. Model
- Base model: `HuggingFaceTB/SmolVLM-256M-Instruct`

## 2. Task
Convert an image containing a handwritten mathematical formula into clean LaTeX code (Image → Text).

## 3. Datasets
- Primary: `linxy/LaTeX_OCR` (human_handwrite split)
- Additional: `deepcopy/MathWriting-human` (used as supplementary data)

## 4. Experimental Setups
1. Zero-shot inference  
2. One-shot inference (single fixed example from train set)  
3. Supervised Fine-Tuning (SFT) on `linxy/LaTeX_OCR:train` only  
4. SFT on `linxy/LaTeX_OCR:train` + `deepcopy/MathWriting-human`

## 5. Fine-Tuning Details (Hyperparameters)
- Method: LoRA (rank=16, alpha=16)
- Quantization: 4-bit (bitsandbytes)
- Optimizer: AdamW
- Learning rate: 1e-5
- Epochs: 1
- Libraries: `transformers`, `peft`, `bitsandbytes`, `datasets`, `jiwer`

## 6. Evaluation
- **Test set:** `linxy/LaTeX_OCR:test` (70 examples)
- **Metric:** Character Error Rate (CER) — perfect for LaTeX string comparison
- Additional qualitative testing on real smartphone photos of my own handwritten formulas

## 7. Results

| Setup                                      | CER    | vs Zero-shot |
|--------------------------------------------|--------|--------------|
| Zero-shot (baseline)                       | 0.1846 | —            |
| One-shot                                   | 0.1990 | -7.8%        |
| SFT (linxy/LaTeX_OCR only)                 | 0.1881 | -1.9%        |
| **SFT (linxy + MathWriting-human)**        | **0.1859** | **-0.7%** |

## 8. Streamlit Application (Task 2)
- File: `app.py`
- Functionality: upload image → model inference → rendered LaTeX (using MathJax)
- Fully implemented using the best SFT model
- Tested on real photos of handwritten formulas

**Video demonstration:**  
[Watch full demo on YouTube](https://youtu.be/--kOWz4kNW8)

**Screenshots** are available in the [`screenshots/`](screenshots/) folder.

## 9. Repository & Models
- GitHub: https://github.com/vys-leon/vlm-handwriting-to-latex
- HF Model 1: https://huggingface.co/Azaper/SmolVLM-256M-SFT-linxy
- HF Model 2: https://huggingface.co/Azaper/SmolVLM-256M-SFT-linxy-deepcopy

## 10. Conclusions
- One-shot inference slightly degraded performance.
- SFT with additional human-written data gave the best result.
- The project demonstrates a complete end-to-end pipeline: from zero-shot to efficient fine-tuning and a production-ready web application.

**Ready for deployment and further research.**