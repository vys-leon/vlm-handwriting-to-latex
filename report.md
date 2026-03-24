# Технический отчёт: Handwritten Formula → LaTeX

**Модель:** HuggingFaceTB/SmolVLM-256M Instruct (https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct)  
**Задача:** Преобразование изображения рукописной математической формулы в LaTeX-код  
**Датасеты:** linxy/LaTeX_OCR (human_handwrite split) + deepcopy/MathWriting-human

## Экспериментальные сетапы

1. **Zero-shot inference**
2. **One-shot inference** (фиксированный пример из train)
3. **Supervised Fine-Tuning (SFT)** — только linxy/LaTeX_OCR:train
4. **SFT + MathWriting-human**

## Метрика оценки
**Character Error Rate (CER)** — основная метрика (хорошо подходит для LaTeX-кода).  

## Результаты

| Setup                        | CER      | Улучшение (по сравнению с zero-shot) |
|------------------------------|----------|---------|
| Zero-shot (baseline)                    | 0.1846    | —       |
| One-shot                     | 0.1990    | ухудшение на 7.8 %       |
| SFT (linxy/LaTeX_OCR)        | 0.1881 | ухудшение на 1.9 % |
| SFT + deepcopy/MathWriting-human | 0.1859 | ухудшение на 0.7 %                         |

**Тестирование на реальной фотографии** — проведено на собственной рукописной формуле.

## Технологии и детали обучения
- **Модель:** SmolVLM-256M-Instruct
- **Fine-tuning:** LoRA (r=16, alpha=16)
- **Оптимизатор:** AdamW, lr=1e-5
- **Библиотеки:** transformers, datasets, peft, jiwer, sacrebleu

## Выводы
- One-shot вышло по метрике хуже, чем zero-shot
- результаты SFT: 
- Проект демонстрирует полный цикл: от zero-shot до fine-tuning и деплоя.

## Ссылки
- GitHub: https://github.com/vys-leon/vlm-handwriting-to-latex
- HF-модель (после SFT using linxy/LaTeX_OCR:train): https://huggingface.co/Azaper/SmolVLM-256M-SFT-linxy 
- HF-модель (после SFT using linxy/LaTeX_OCR:train + deepcopy/MathWriting-human): https://huggingface.co/Azaper/SmolVLM-256M-SFT-linxy-deepcopy

**Дата:** 24 марта 2026