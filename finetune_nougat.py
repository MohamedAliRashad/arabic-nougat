# This script is used to fine-tune the Nougat model on the Arabic image to markdown dataset.
# The dataset is loaded using the datasets library and split into training and testing datasets.
# The model is then fine-tuned using the Seq2SeqTrainer class.
# The model is saved after training.
# The script will benefit from a machine with multiple GPUs to run efficiently.

from transformers import (
    NougatProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
)
from datasets import load_dataset
import torch
from pathlib import Path
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
import warnings

warnings.filterwarnings("ignore")

# accelerate launch --multi_gpu --num_processes 4 finetune_nougat.py

processor = NougatProcessor.from_pretrained("facebook/nougat-base")
model = VisionEncoderDecoderModel.from_pretrained(
    "facebook/nougat-base",
    torch_dtype=torch.bfloat16,
    attn_implementation={"decoder": "flash_attention_2", "encoder": "eager"},
)
context_length = model.decoder.config.max_position_embeddings
torch_dtype = model.dtype
print(f"Context length: {context_length}")
print(f"Model dtype: {torch_dtype}")
print(model.decoder)

# Measure model size and number of parameters
model_size = sum(p.numel() for p in model.parameters()) / 1e9
print(f"Model size: {model_size}")

dataset = load_dataset("MohamedRashad/arabic-img2md")
dataset = dataset.train_test_split(test_size=0.1)
print(dataset)


class DataCollatorVisionSeq2SeqWithPadding:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, samples):
        batch = {}
        pil_images = [sample["image"] for sample in samples]
        batch["pixel_values"] = self.processor.image_processor(
            pil_images, return_tensors="pt"
        ).pixel_values.to(torch_dtype)

        markdowns = [sample["markdown"] for sample in samples]
        tokenized_inputs = self.processor.tokenizer(
            markdowns,
            return_tensors="pt",
            padding="max_length",
            max_length=context_length,
            truncation=True,
        )

        labels = tokenized_inputs["input_ids"]
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        batch["labels"] = labels
        batch["decoder_attention_mask"] = tokenized_inputs["attention_mask"]

        return batch


data_collator = DataCollatorVisionSeq2SeqWithPadding(processor)
model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id

training_args = Seq2SeqTrainingArguments(
    output_dir=str(Path(__file__).parent / "arabic_base_nougat_logs"),
    num_train_epochs=100,
    predict_with_generate=True,
    eval_strategy="steps",
    save_strategy="steps",
    learning_rate=1e-4,
    report_to="none",
    save_steps=100,
    eval_steps=100,
    gradient_checkpointing=True,  # Great for memory saving
    gradient_checkpointing_kwargs={"use_reentrant": False},  # To remove the warning
    per_device_train_batch_size=3,
    gradient_accumulation_steps=6,
    per_device_eval_batch_size=3,
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
trainer.train(resume_from_checkpoint=False)

# Save model
trainer.save_model(str(Path(__file__).parent / "arabic-base-nougat3"))
processor.save_pretrained(str(Path(__file__).parent / "arabic-base-nougat3"))
print("Model saved!")
