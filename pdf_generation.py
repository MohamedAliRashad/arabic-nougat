from datasets import load_dataset
import multiprocessing
from bs4 import BeautifulSoup
import weasyprint
import html2text
from pathlib import Path
import random
import pandas as pd
import shutil
from tqdm.auto import tqdm
from pdf2image import convert_from_bytes
from transformers import AutoTokenizer
import csv
from filelock import FileLock
import os
import io
from colorama import Fore

def remove_lock_file(file_path):
    lock_file = f"{file_path}.lock"
    if os.path.exists(lock_file):
        os.remove(lock_file)

def html_to_markdown(html_content):
    soup = BeautifulSoup(html_content, 'lxml')
    
    tag_to_token = {'img': '[IMAGE]'}
    tags_to_ignore = ['svg', 'defs', 'g', 'symbol', 'path', 'use', 'eqsvg', 
                      'svg:svg', 'svg:metadata', 'svg:defs', 'svg:g', 'svg:path']

    for tag in soup.find_all(list(tag_to_token.keys()) + tags_to_ignore):
        if tag.name in tag_to_token:
            tag.replace_with(tag_to_token[tag.name])
        else:
            tag.decompose()

    h = html2text.HTML2Text()
    h.ignore_links = True
    h.ignore_emphasis = True
    h.body_width = 0
    
    return h.handle(str(soup))

def process_html(args):
    idx, html_content = args

    soup = BeautifulSoup(html_content, 'lxml')
    all_elements = soup.find_all(recursive=True)
    num_elements = int(0.2*len(all_elements))
    all_elements = random.sample(all_elements, num_elements)

    for jdx, element in enumerate(all_elements):
        random_width, random_height = random.choice(available_target_size)
        if random.random() < 0.05:
            random_width, random_height = random_height, random_width
        
        html_to_render = html_text.format(
            html_content_ar=str(element),
            font=random.choice(available_fonts),
            font_size=random.choice(available_font_sizes),
            font_weight=random.choice(available_font_weights),
            width=random_width,
            height=random_height,
            num_columns = 2 if random.random() < 0.05 else 1
        )
        
        try:
            html = weasyprint.HTML(string=html_to_render)
            document = html.render()
        except ZeroDivisionError as e:
            print(f"Error processing HTML at index {idx}, skipping...")
            continue

        if len(document.pages) > 1:
            # print(f"Document has more than one page, skipping...")
            continue

        # Instead of writing to disk, we'll use BytesIO
        pdf_bytes = io.BytesIO()
        document.write_pdf(pdf_bytes)
        pdf_bytes.seek(0)

        markdown = html_to_markdown(str(element)).strip()
        num_tokens = len(tokenizer.tokenize(markdown, add_special_tokens=False))
        if num_tokens > 8192 or num_tokens < 1024:
            # print(f"Markdown content is too short, skipping...")
            continue

        # Convert PDF bytes to image
        images = convert_from_bytes(pdf_bytes.getvalue())
        image_path = dataset_output_path / f"{str(idx).zfill(5)}_{jdx}.png"
        images[0].save(image_path, 'PNG')

        # Append to metadata.csv with lock
        metadata = [image_path.name, markdown]
        with FileLock(f"{metadata_file}.lock"):
            with open(metadata_file, "a", newline='', encoding='utf-8') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(metadata)

    print(f"Processed book number {Fore.GREEN}{idx}{Fore.RESET}")

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("MohamedRashad/arabic-large-nougat")
    # tokenizer = AutoTokenizer.from_pretrained("facebook/nougat-base")

    available_target_size = [
        (672, 896),   # Base Nougat model size
        (595, 842),   # A4 (most common for Arabic books)
        (612, 792),   # US Letter
        (516, 729),   # B5 (common for books)
        (672, 896),   # Base Nougat model size
    ]
    available_fonts = [
        "Amiri",              # Very common in academic books
        "Noto Naskh Arabic",  # Excellent readability
        "Cairo",             # Modern and clear
        "IBM Plex Sans Arabic", # Professional looking
        "Almarai",           # Clear and modern
        "Tajawal",           # Good readability
        "Noto Kufi Arabic",  # Good for headers
    ]

    available_font_sizes = [14, 16, 18, 20]  # More reasonable sizes for Arabic text
    available_font_weights = [400, 500, 600]  # Reduced weights for better clarity
    dataset_output_path = Path(__file__).parent / "hindawi_dataset_eval"
    if dataset_output_path.exists():
        shutil.rmtree(dataset_output_path)
    dataset_output_path.mkdir(exist_ok=True)

    metadata_file = dataset_output_path / "metadata.csv"

    # Write CSV header
    with open(metadata_file, "w", newline='', encoding='utf-8') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["file_name", "markdown"])

    html_text = """<!DOCTYPE html>
<html dir="rtl" lang="ar">
<head>
    <meta charset="utf-8">
    <style>
        body {{
            font-family: '{font}', sans-serif;
            font-size: {font_size}px;
            column-count: {num_columns};
            column-gap: 20px;
            column-fill: auto;
            height: 100%;
            line-height: 1.6;
        }}
        h1 {{
            column-span: all;
            page-break-before: avoid;
            page-break-after: avoid;
            margin-bottom: 12px;
        }}
        p {{
            text-align: justify;
            page-break-inside: avoid;
            margin-bottom: 12px;
        }}
        .center {{
            text-align: center;
        }}
        .align_left {{
            text-align: left;
        }}
        .no-page-break {{
            page-break-inside: avoid;
        }}
        @page {{
            size: {width}px {height}px;
            margin: 1in;
        }}
    </style>
    <link href="https://fonts.googleapis.com/css2?family={font}:wght@{font_weight}&display=swap" rel="stylesheet">
</head>

<body>
{html_content_ar}
</body>
</html>
"""

    # Login using e.g. `huggingface-cli login` to access this dataset
    ds = load_dataset("MohamedRashad/hindawi-dataset", split="train")
    df = pd.DataFrame(ds)
    df = df.sample(10)
    print(f"Total records: {len(df)}")

    # Use multiprocessing to parallelize the processing with a progress bar
    with multiprocessing.Pool(128) as pool:
        list(tqdm(pool.imap(process_html, enumerate(df["html_content"])), total=len(df)))

    # Remove lock file
    remove_lock_file(metadata_file)
    
    print("Processing complete!")