# Qwen2.5-1.5B MMLU Evaluation

A quick evaluation script for testing **Qwen/Qwen2.5-1.5B** on the **MMLU high school geography** benchmark using zero-shot multiple-choice prompting.

---

## Overview

This project:

- Loads the `cais/mmlu` dataset (high_school_geography split) and samples 100 test examples.
- Loads the **Qwen2.5-1.5B** causal language model and its tokenizer.
- Builds a simple prompt for each question listing choices A–D.
- Compares the model's next-token logits for the letters `A`, `B`, `C`, `D` and picks the highest.
- Reports accuracy across the 100 sampled questions.
- Demonstrates tokenizer behavior across languages (English vs. Sinhala).

---

## Requirements

- Python 3.9+
- A CUDA-capable GPU is recommended (the script uses `device_map="auto"` and `float16`).

### Dependencies

- `transformers`
- `datasets`
- `accelerate`
- `torch`

Install with:

```bash
pip install -q transformers datasets accelerate torch