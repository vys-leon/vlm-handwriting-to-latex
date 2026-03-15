# Handwritten Formula → LaTeX (SmolVLM-256M)

**Статус проекта (15 марта 2026):**  
Zero-shot и One-shot inference готовы + метрики посчитаны.  
SFT (Supervised Fine-Tuning) в процессе — обновлю до 24 марта.

**Project** для демонстрации fine-tuning мультимодальной VLM модели.

Преобразует рукописные математические формулы в LaTeX-код.

## Результаты

| Setup                        | CER      | Улучшение |
|------------------------------|----------|---------|
| Zero-shot                    | 0.1752    | —       |
| One-shot                     | 0.1705    | улучшение на 2.68%       |
| SFT (linxy/LaTeX_OCR)        | **0.XXXX** | **XX%** |
| SFT + MathWriting | **0.XXXX** | **XX%**                         |

**Метрика**: Character Error Rate (CER) — оптимально для LaTeX-кода.

## Что сделано
- Zero-shot и One-shot inference
- Supervised Fine-Tuning (LoRA + 4-bit)
- Оценка на тестовом сете (70 примеров)
- Тестирование на **реальной фотографии** собственной формулы

## Ссылки
- **Hugging Face модель**: 
- **Технический отчёт**: [report.md](report.md)

## Как запустить
```bash
git clone https://github.com/vys-leon/vlm-handwriting-to-latex.git
cd vlm-handwriting-to-latex
pip install -r requirements.txt
python src/inference.py
