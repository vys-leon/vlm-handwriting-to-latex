# Handwritten Formula → LaTeX (SmolVLM-256M)

**Project** для демонстрации fine-tuning мультимодальной VLM модели.

Преобразует рукописные математические формулы в LaTeX-код.

## Результаты

| Setup                        | CER      | Улучшение (по сравнению с zero-shot) |
|------------------------------|----------|---------|
| Zero-shot (baseline)                    | 0.1846    | —       |
| One-shot                     | 0.1990    | ухудшение на 7.8 %       |
| SFT (linxy/LaTeX_OCR)        | 0.1881 | ухудшение на 1.9 % |
| SFT + deepcopy/MathWriting-human | 0.1859 | ухудшение на 0.7 %                         |

**Метрика**: Character Error Rate (CER) — оптимально для LaTeX-кода.

## Что сделано
- Zero-shot и One-shot inference
- Supervised Fine-Tuning (LoRA + 4-bit) на датасете linxy/LaTeX_OCR
- Supervised Fine-Tuning на датасетах linxy/LaTeX_OCR + deepcopy/MathWriting-human
- Оценка на тестовом сете (linxy/LaTeX_OCR:test; 70 примеров)
- Тестирование на **реальных фотографиях** собственных формул

## Ссылки
- **Hugging Face модель (SFT using linxy/LaTeX_OCR:train)**: https://huggingface.co/Azaper/SmolVLM-256M-SFT-linxy
- **Hugging Face модель (SFT using linxy/LaTeX_OCR:train + deepcopy/MathWriting-human)**: https://huggingface.co/Azaper/SmolVLM-256M-SFT-linxy-deepcopy
- **Технический отчёт**: [report.md](report.md)

## Как запустить
```bash
git clone https://github.com/vys-leon/vlm-handwriting-to-latex.git
cd vlm-handwriting-to-latex
pip install -r requirements.txt
python src/inference.py
