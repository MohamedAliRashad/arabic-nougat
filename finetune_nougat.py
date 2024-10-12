import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

from transformers import (
    NougatProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
    AutoTokenizer,
    MBartForCausalLM,
    DonutSwinModel,
)
from tokenizers.processors import TemplateProcessing
from datasets import load_dataset
import torch
from pathlib import Path
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import warnings
warnings.filterwarnings("ignore")

# accelerate launch --multi_gpu --num_processes 4 finetune_nougat.py

processor = NougatProcessor.from_pretrained(Path(__file__).parent / "arabic-nougat")
processor.tokenizer._tokenizer.post_processor = TemplateProcessing(
    single="<s> $A </s>",
    special_tokens=[("<s>", processor.tokenizer.bos_token_id), ("</s>", processor.tokenizer.eos_token_id)],
)
model = VisionEncoderDecoderModel.from_pretrained(
    Path(__file__).parent / "arabic-nougat", torch_dtype=torch.bfloat16
)
decoder = MBartForCausalLM._from_config(model.decoder.config, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
decoder.load_state_dict(model.decoder.state_dict())
model.decoder = decoder
config = model.decoder.config
print(config)

# Measure model size and number of parameters
model_size = sum(p.numel() for p in model.parameters()) / 1e9
print(f"Model size: {model_size}")

dataset_path = Path(__file__).parent / "hindawi_dataset2"
dataset = load_dataset("imagefolder", data_dir=dataset_path, split="train")
dataset = dataset.shuffle(seed=42)
dataset = dataset.train_test_split(test_size=0.1)
print(dataset)

class DataCollatorVisionSeq2SeqWithPadding:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, samples):
        batch = {}
        pil_images = [sample["image"] for sample in samples]
        batch["pixel_values"] = self.processor.image_processor(pil_images, return_tensors="pt").pixel_values.to(torch.bfloat16)
        
        markdowns = [sample["markdown"] for sample in samples]
        tokenized_inputs = self.processor.tokenizer(markdowns, return_tensors="pt", padding="max_length", max_length=config.max_position_embeddings, truncation=True)
        
        labels = tokenized_inputs['input_ids']
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        batch["labels"] = labels
        batch["decoder_attention_mask"] = tokenized_inputs["attention_mask"]

        return batch

data_collator = DataCollatorVisionSeq2SeqWithPadding(processor)
model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id

training_args = Seq2SeqTrainingArguments(
    output_dir=str(Path(__file__).parent / "arabic_nougat_logs"),
    num_train_epochs=10,
    predict_with_generate=True,
    eval_strategy="steps",
    save_strategy="steps",
    learning_rate=5e-4,
    report_to="none",
    save_steps=300,
    eval_steps=300,
    gradient_checkpointing=True,  # Great for memory saving
    gradient_checkpointing_kwargs={"use_reentrant": False},  # To remove the warning
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    per_device_eval_batch_size=2,
    eval_accumulation_steps=8,
    overwrite_output_dir=True,
    save_total_limit=1,
    bf16=True,
    logging_steps=5,
    dataloader_num_workers=8,  # Use multiple processes for data loading
    dataloader_pin_memory=True,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    ddp_find_unused_parameters=False,  # Don't know what this does
    remove_unused_columns=False,  # Gave me warnings so I set it to False
)

trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=processor.image_processor,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

# Train the model
trainer.train(resume_from_checkpoint=True)

# Save model
trainer.save_model(str(Path(__file__).parent / "arabic-nougat2"))
processor.save_pretrained(str(Path(__file__).parent / "arabic-nougat2"))
print("Model saved!")
