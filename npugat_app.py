from PIL import Image
import torch
from transformers import NougatProcessor, VisionEncoderDecoderModel, MBartForCausalLM
from pathlib import Path
from bidi.algorithm import get_display
import arabic_reshaper

# Load the model and processor
processor = NougatProcessor.from_pretrained(Path(__file__).parent / "arabic-nougat")
model = VisionEncoderDecoderModel.from_pretrained(
    Path(__file__).parent / "arabic-nougat", torch_dtype=torch.bfloat16
)
decoder = MBartForCausalLM._from_config(model.decoder.config, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
decoder.load_state_dict(model.decoder.state_dict())
model.decoder = decoder

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

context_length = 8192

def predict(img_path):
    # prepare PDF image for the model
    image = Image.open(img_path)
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(torch.bfloat16)

    # generate transcription
    outputs = model.generate(
        pixel_values.to(device),
        min_length=1,
        max_new_tokens=context_length,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        eos_token_id=processor.tokenizer.eos_token_id,
    )

    page_sequence = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    page_sequence = get_display(arabic_reshaper.reshape(page_sequence))
    return page_sequence

print(predict(Path(__file__).parent / "book_page.jpeg"))
