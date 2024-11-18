# Requires a machine with multiple GPUs to run efficiently

from transformers import NougatProcessor, VisionEncoderDecoderModel
import torch
from datasets import load_dataset, Dataset, features
from pathlib import Path
from tqdm import tqdm
import evaluate
import re
from tabulate import tabulate

models_supported = {
    "nougat-small": [
        NougatProcessor.from_pretrained("facebook/nougat-small"),
        VisionEncoderDecoderModel.from_pretrained("facebook/nougat-small").to("cuda"),
    ],
    "nougat-base": [
        NougatProcessor.from_pretrained("facebook/nougat-base"),
        VisionEncoderDecoderModel.from_pretrained("facebook/nougat-base").to("cuda"),
    ],
    "arabic-small-nougat": [
        NougatProcessor.from_pretrained("MohamedRashad/arabic-small-nougat"),
        VisionEncoderDecoderModel.from_pretrained("MohamedRashad/arabic-small-nougat").to("cuda"),
    ],
    "arabic-base-nougat": [
        NougatProcessor.from_pretrained("MohamedRashad/arabic-base-nougat"),
        VisionEncoderDecoderModel.from_pretrained(
            "MohamedRashad/arabic-base-nougat",
            torch_dtype=torch.bfloat16,
            attn_implementation={"decoder": "flash_attention_2", "encoder": "eager"},
        ).to("cuda"),
    ],
    "arabic-large-nougat": [
        NougatProcessor.from_pretrained("MohamedRashad/arabic-large-nougat"),
        VisionEncoderDecoderModel.from_pretrained(
            "MohamedRashad/arabic-large-nougat",
            torch_dtype=torch.bfloat16,
            attn_implementation={"decoder": "flash_attention_2", "encoder": "eager"},
        ).to("cuda"),
    ],
}


# Load dataset
test_dataset = load_dataset("MohamedRashad/arabic-img2md")
test_dataset = test_dataset.shuffle(seed=42).select(range(100 * 16))

def process_batch(batch, model_name):
    processor = models_supported[model_name][0]
    model = models_supported[model_name][1]
    context_length = model.decoder.config.max_position_embeddings

    pixel_values = processor(batch['image'], return_tensors="pt").pixel_values.to(model.dtype).to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(pixel_values, max_new_tokens=context_length, bad_words_ids=[[processor.tokenizer.unk_token_id]], repetition_penalty=1.5)
    predictions = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return predictions

# Generate predictions
results = []
for model_name in models_supported.keys():
    batch_size = 16  # Increase batch size to utilize all GPUs
    predictions = []
    references = []
    for i in tqdm(range(0, len(test_dataset), batch_size)):
        batch = test_dataset[i:i+batch_size]
        predictions.extend(process_batch(batch, model_name))
        references.extend(batch['markdown'])

    # Calculate metrics
    bleu = evaluate.load("bleu")
    cer = evaluate.load("cer")
    wer = evaluate.load("wer")

    bleu_score = bleu.compute(predictions=predictions, references=[[r] for r in references])['bleu']
    cer_score = cer.compute(predictions=predictions, references=references)
    wer_score = wer.compute(predictions=predictions, references=references)

    # Calculate exact match accuracy
    exact_matches = sum(pred == ref for pred, ref in zip(predictions, references))
    exact_match_accuracy = exact_matches / len(predictions)

    # Markdown structure accuracy
    def markdown_structure_accuracy(pred, ref):
        # Simple function to compare basic markdown elements
        pred_headers = len(re.findall(r'^#+\s', pred, re.MULTILINE))
        ref_headers = len(re.findall(r'^#+\s', ref, re.MULTILINE))
        pred_lists = len(re.findall(r'^\s*[-*+]\s', pred, re.MULTILINE))
        ref_lists = len(re.findall(r'^\s*[-*+]\s', ref, re.MULTILINE))
        
        header_accuracy = 1 - abs(pred_headers - ref_headers) / max(ref_headers, 1)
        list_accuracy = 1 - abs(pred_lists - ref_lists) / max(ref_lists, 1)
        
        return (header_accuracy + list_accuracy) / 2

    structure_scores = [markdown_structure_accuracy(p, r) for p, r in zip(predictions, references)]
    avg_structure_score = sum(structure_scores) / len(structure_scores)

    # # Optional: Weighted Composite Score
    # composite_score = (
    #     0.25 * bleu_score +
    #     0.25 * (1 - cer_score) +  # CER is better when lower
    #     0.25 * (1 - wer_score) +  # WER is better when lower
    #     0.15 * exact_match_accuracy +
    #     0.10 * avg_structure_score
    # )

    # Inside your model loop, replace the print statements with:
    model_results = {
        "Model": model_name,
        "BLEU Score": f"{bleu_score:.4f}",
        "CER": f"{cer_score:.4f}",
        "WER": f"{wer_score:.4f}",
        # "Exact Match": f"{exact_match_accuracy:.4f}",
        "Structure Accuracy": f"{avg_structure_score:.4f}",
        # "Composite Score": f"{composite_score:.4f}"
    }
    results.append(model_results)

# After the model loop, print the combined table:
headers = list(results[0].keys())
table_data = [[result[col] for col in headers] for result in results]
print("\nComparative Results:")
print(tabulate(table_data, headers=headers, tablefmt="grid"))
print()
