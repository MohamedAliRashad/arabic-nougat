from transformers import AutoTokenizer
from datasets import load_dataset
from pathlib import Path

tokenizer1 = AutoTokenizer.from_pretrained("MohamedRashad/arabic-large-nougat")
tokenizer2 = AutoTokenizer.from_pretrained("MohamedRashad/arabic-base-nougat")

dataset_path1 = Path(__file__).parent / "hindawi_dataset_large"
dataset = load_dataset("imagefolder", data_dir=dataset_path1, split="train")
prompts = dataset["markdown"][:5000]

ratios = []
for prompt in prompts:
    tokens1 = tokenizer1.tokenize(prompt)
    tokens2 = tokenizer2.tokenize(prompt)
    ratios += [len(tokens2) / len(tokens1)]
    
print(sum(ratios) / len(ratios))