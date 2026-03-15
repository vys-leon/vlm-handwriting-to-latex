# Технический отчёт: Handwritten Formula → LaTeX

**Статус проекта (15 марта 2026):**  
Zero-shot и One-shot inference готовы + метрики посчитаны.  
SFT (Supervised Fine-Tuning) в процессе — обновлю до 24 марта.

**Модель:** SmolVLM-256M-Instruct (HuggingFaceTB)  
**Задача:** Преобразование изображения рукописной математической формулы в LaTeX-код  
**Датасет:** linxy/LaTeX_OCR (human_handwrite split)

## Экспериментальные сетапы

1. **Zero-shot inference**
2. **One-shot inference** (фиксированный пример из train)
3. **Supervised Fine-Tuning (SFT)** — только linxy/LaTeX_OCR:train
4. **SFT + MathWriting-human**

## Метрика оценки
**Character Error Rate (CER)** — основная метрика (хорошо подходит для LaTeX-кода).  
Дополнительно считались BLEU и ROUGE-L.

## Результаты

| Setup                          | CER     | Улучшение относительно zero-shot |
|--------------------------------|---------|----------------------------------|
| Zero-shot                      | 0.1752   | —                                |
| One-shot                       | 0.1705  | улучшение на 2.68%                            |
| SFT (linxy/LaTeX_OCR)          | **0.XXXX** | **XX%**                         |
| SFT + MathWriting | **0.XXXX** | **XX%**                         |

**Тестирование на реальной фотографии** — проведено на собственной рукописной формуле.

## Технологии и детали обучения
- **Модель:** SmolVLM-256M-Instruct
- **Квантизация:** 4-bit (BitsAndBytes)
- **Fine-tuning:** LoRA (r=8, alpha=8, target_modules=все linear слои)
- **Оптимизатор:** AdamW, lr=1e-4
- **Библиотеки:** transformers, datasets, peft, jiwer, sacrebleu

## Выводы
- Zero-shot и one-shot дают примерно одинаковое качество (модель слабо реагирует на один пример).
- результаты SFT (SFT пока в процессе)
- Проект демонстрирует полный цикл: от zero-shot до fine-tuning и деплоя.

## Ссылки
- GitHub: https://github.com/vys-leon/vlm-handwriting-to-latex
- HF-модель (после SFT): https://huggingface.co/Azaper/smolvlm-latex-ocr

**Дата:** 15 марта 2026