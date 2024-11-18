# Arabic-Nougat

Arabic-Nougat is a suite of Optical Character Recognition (OCR) models designed to extract structured text in Markdown format from Arabic book pages. This repository provides tools for fine-tuning, evaluation, dataset preparation, and tokenizer analysis for the Arabic-Nougat models, which build on Meta’s Nougat architecture.
<p align="center">
  <img src="https://github.com/user-attachments/assets/028a77b4-8ca3-4e21-a3f9-84b1999550e6" width="60%">
</p>


## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Evaluate Models](#evaluate-models)
  - [Fine-Tune Models](#fine-tune-models)
  - [Generate Synthetic PDFs and Markdown](#generate-synthetic-pdfs-and-markdown)
  - [Tokenizer Analysis](#tokenizer-analysis)
  - [Quick Test](#quick-test)
- [Models](#models)
- [Dataset](#dataset)
- [License](#license)

---

## Overview

Arabic-Nougat is tailored to process Arabic text, handling the unique challenges of the script, such as its cursive nature and contextual letter forms. It extends Meta’s Nougat OCR model with custom enhancements:
- **Advanced Tokenization**: Includes the `Aranizer-PBE-86k` tokenizer, optimized for Arabic text.
- **Extended Context Length**: Supports up to 8192 tokens, suitable for processing lengthy documents.
- **Dataset**: Uses the synthetic `arabic-img2md` dataset, designed to train models for Markdown extraction.

---

## Features

- Fine-tune the model on custom datasets.
- Evaluate models using standard metrics like BLEU, CER, and WER.
- Generate synthetic PDFs and Markdown from HTML for dataset creation.
- Analyze tokenizer performance and efficiency.
- Pretrained models: `arabic-small-nougat`, `arabic-base-nougat`, and `arabic-large-nougat`.

---

## Repository Structure

```
├── eval_model.py          # Evaluate Arabic-Nougat models
├── finetune_nougat.py     # Fine-tune the Nougat model
├── pdf_generation.py      # Generate synthetic PDFs and Markdown
├── tokenizer_ratios.py    # Analyze tokenizer performance
├── try_nougat.py          # Test the model on a sample image
```

---

## Installation

### Prerequisites
1. Python 3.8 or higher.
2. A machine with multiple GPUs for fine-tuning and evaluation.
3. Install the required Python libraries:
   ```bash
   pip install transformers datasets evaluate weasyprint html2text pdf2image pillow tabulate bidi arabic-reshaper colorama filelock tqdm
   ```

4. Install system dependencies for `weasyprint`:
   ```bash
   sudo apt install libpango1.0-dev libpangocairo-1.0-0
   ```

5. Ensure that you have `huggingface-cli` installed and logged in:
   ```bash
   pip install huggingface-hub
   huggingface-cli login
   ```

---

## Usage

### Evaluate Models

Evaluate the performance of Arabic-Nougat models on a dataset
```bash
python eval_model.py
```
- Metrics calculated:
  - BLEU
  - Character Error Rate (CER)
  - Word Error Rate (WER)
  - Markdown Structure Accuracy (custom metric, check the paper to learn more)

**Note:** Won't produce the same numbers present in the paper as the eval dataset is not open.

### Fine-Tune Models

Fine-tune the Nougat model on the `arabic-img2md` dataset:
```bash
accelerate launch --multi_gpu --num_processes 4 finetune_nougat.py
```
- Configurations:
  - Context length: 8192 tokens.
  - Data collators and efficient gradient accumulation.

### Generate Synthetic PDFs and Markdown

Create synthetic datasets from HTML content:
```bash
python pdf_generation.py
```
- Converts HTML to PDFs and Markdown for training datasets.
- Configurable fonts, sizes, and page layouts for diversity.

### Tokenizer Analysis

Compare tokenization efficiency between different tokenizers:
```bash
python tokenizer_ratios.py
```
- Outputs the average tokenization ratio between models like `arabic-large-nougat` and `arabic-base-nougat`.

### Quick Test

Test the model on a sample image:
```bash
python try_nougat.py
```
- Input: Path to an image of a book page (there is a default value in the script).
- Output: Extracted Markdown text.

---

## Models

The following pretrained models are available:
1. [arabic-small-nougat](https://huggingface.co/MohamedRashad/arabic-small-nougat)
   - Fine-tuned from `facebook/nougat-small`.
   - Smaller context length (2048 tokens).
2. [arabic-base-nougat](https://huggingface.co/MohamedRashad/arabic-base-nougat)
   - Fine-tuned from `facebook/nougat-base`.
   - Context length: 4096 tokens.
3. [arabic-large-nougat](https://huggingface.co/MohamedRashad/arabic-large-nougat)
   - Built with `Aranizer-PBE-86k` tokenizer.
   - Extended context length: 8192 tokens.

Access the models on [Hugging Face](https://huggingface.co/collections/MohamedRashad/arabic-nougat-673a3f540bd92904c9b92a8e).

---

## Dataset

### Arabic-Img2MD
- A synthetic dataset of 13.7k samples containing:
  - **Input**: Images of Arabic book pages.
  - **Output**: Corresponding Markdown text.

Dataset available on [Hugging Face](https://huggingface.co/datasets/MohamedRashad/arabic-img2md).

---

## License

This project is released under the [Creative Commons Attribution-ShareAlike (CC BY-SA)](https://creativecommons.org/licenses/by-sa/4.0/) license. Feel free to use, share, and adapt the work, provided proper attribution is given and adaptations are shared under the same terms.
