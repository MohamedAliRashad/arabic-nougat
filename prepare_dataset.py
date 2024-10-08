from pathlib import Path
from transformers import AutoTokenizer, NougatProcessor
import pandas as pd
from tqdm import tqdm
from PIL import Image
import random

tokenizer = AutoTokenizer.from_pretrained(
    "riotu-lab/Aranizer-PBE-86k",
    use_fast=True,
    bos_token="<s>",
    eos_token="</s>",
    pad_token="<pad>",
    unk_token="<unk>",
    additional_special_tokens=["[IMAGE]"],
)

image_processor = NougatProcessor.from_pretrained("facebook/nougat-base").image_processor
size = image_processor.size
nougat_height, nougat_width = size["height"], size["width"]

dataset_path = Path(__file__).parent / 'hindawi_dataset'
images_paths = list(dataset_path.glob('*.png'))
print(images_paths[0])

book_ids = list(set([path.stem.split('_')[0] for path in images_paths]))
print(len(book_ids))

nougat_dataset_folder = Path(__file__).parent / 'nougat_dataset'
nougat_dataset_folder.mkdir(exist_ok=True)

df = pd.DataFrame(columns=['file_name', 'markdown'])
for book_id in tqdm(book_ids):
    book_images = list(dataset_path.glob(f"{book_id}_*.png"))
    book_images = random.sample(book_images, min(5, len(book_images)))
    for image_path in book_images:
        markdown_path = image_path.parent / f"{image_path.stem}.md"
        if not markdown_path.exists():
            continue
        
        try:
            pil_image = Image.open(image_path)
            resized_image = pil_image.resize((nougat_width, nougat_height))
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            continue

        resized_image.save(nougat_dataset_folder / image_path.name, 'PNG')
        with open(markdown_path, 'r') as f:
            markdown_content = f.read()
        df = df._append({'file_name': image_path.name, 'markdown': markdown_content}, ignore_index=True)
        
    df.to_csv(nougat_dataset_folder / 'metadata.csv', index=False)
