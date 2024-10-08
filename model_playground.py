from transformers import (
    NougatProcessor,
    VisionEncoderDecoderModel,
    MBartForCausalLM,
    AutoTokenizer,
)
import torch
from PIL import Image
from pathlib import Path

processor = NougatProcessor.from_pretrained("facebook/nougat-base")
model = VisionEncoderDecoderModel.from_pretrained("facebook/nougat-base", torch_dtype=torch.bfloat16)
print(model.config)
tokenizer = AutoTokenizer.from_pretrained(
    "riotu-lab/Aranizer-PBE-86k",
    use_fast=True,
    bos_token="<s>",
    eos_token="</s>",
    pad_token="<pad>",
    unk_token="<unk>",
    additional_special_tokens=["[IMAGE]"],
)
processor.tokenizer = tokenizer
config = model.decoder.config
config.bos_token_id = tokenizer.bos_token_id
config.eos_token_id = tokenizer.eos_token_id
config.max_position_embeddings = 8 * 1024
config.pad_token_id = tokenizer.pad_token_id
config.vocab_size = tokenizer.vocab_size
config.decoder_layers = 10
config.d_model = 1024
config.use_cache = False
processor.tokenizer = tokenizer
model.decoder = MBartForCausalLM._from_config(
    config, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    # config, torch_dtype=torch.bfloat16
    # config
)
model.cuda()

# Measure model size and number of parameters
model_size = sum(p.numel() for p in model.parameters()) / 1e9
print(f"Model size: {model_size}")

dataset_path = Path(__file__).parent / "nougat_dataset"
img_path = list(dataset_path.glob("*.png"))[0]
image = Image.open(img_path)
pixel_values = processor(image, return_tensors="pt").pixel_values.to(torch.bfloat16)

# generate transcription
outputs = model.generate(
    pixel_values.cuda(),
    min_length=1,
    max_new_tokens=2048,
    bad_words_ids=[[processor.tokenizer.unk_token_id]],
)
print(outputs)