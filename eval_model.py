import evaluate
from datasets import load_dataset
from transformers import NougatProcessor, VisionEncoderDecoderModel
import torch
from pathlib import Path
from tqdm import tqdm


# Load model and move to GPU
model_path = "arabic-nougat"
processor = NougatProcessor.from_pretrained(model_path)
model = VisionEncoderDecoderModel.from_pretrained(model_path).cuda()


# Load dataset
dataset_path = Path(__file__).parent / "hindawi_dataset_test"
test_dataset = load_dataset("imagefolder", data_dir=dataset_path, split="train")
test_dataset = test_dataset.shuffle(seed=42).select(range(10*32))

def process_batch(batch):
    pixel_values = processor(batch['image'], return_tensors="pt").pixel_values.to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(pixel_values, max_length=8192)
    predictions = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return predictions

# Generate predictions
batch_size = 32
predictions = []
references = []
for i in tqdm(range(0, len(test_dataset), batch_size)):
    batch = test_dataset[i:i+batch_size]
    predictions.extend(process_batch(batch))
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

# Print results
print(f"BLEU Score: {bleu_score:.4f}")
print(f"Character Error Rate (CER): {cer_score:.4f}")
print(f"Word Error Rate (WER): {wer_score:.4f}")
print(f"Exact Match Accuracy: {exact_match_accuracy:.4f}")

# # Arabic Small Nougat Model Results:
# BLEU Score: 0.6236
# Character Error Rate (CER): 0.4044
# Word Error Rate (WER): 0.3867
# Exact Match Accuracy: 0.0656