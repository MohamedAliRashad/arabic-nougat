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
import PyPDF2
from pdf2image import convert_from_path
from transformers import AutoTokenizer
import csv
from filelock import FileLock
import os

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
    h.ignore_links = False
    h.ignore_images = False
    h.ignore_emphasis = False
    h.body_width = 0
    
    return h.handle(str(soup))

def process_html(args):
    idx, html_content = args

    # Read the HTML content
    soup = BeautifulSoup(html_content, 'lxml')
    all_elements = soup.find_all(recursive=True)

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
        
        pdf_path = dataset_output_path / f"{str(idx).zfill(5)}.pdf"
        html = weasyprint.HTML(string=html_to_render).render()
        html.write_pdf(pdf_path)

        # Open the PDF file in read-binary mode
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            number_of_pages = len(reader.pages)

        if number_of_pages > 1:
            print(f"PDF file {pdf_path} has more than one page, deleting...")
            pdf_path.unlink()
            continue
        else:
            markdown = html_to_markdown(str(element)).strip()
            num_tokens = len(tokenizer.tokenize(markdown, add_special_tokens=False))
            if num_tokens < 5:
                print(f"Markdown content is too short, skipping...")
                pdf_path.unlink()
                continue
            images = convert_from_path(pdf_path)
            image_path = pdf_path.parent / f"{pdf_path.stem}_{jdx}.png"
            images[0].save(image_path, 'PNG')
            pdf_path.unlink()

            # Append to metadata.csv with lock
            metadata = [image_path.name, markdown]
            with FileLock(f"{metadata_file}.lock"):
                with open(metadata_file, "a", newline='', encoding='utf-8') as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerow(metadata)

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(Path(__file__).parent / "arabic-nougat-tokenizer")
    available_target_size = [
        (672, 896),   # Base Nougat model size
        (612, 792),   # US Letter (8.5" x 11")
        (612, 1008),  # US Legal (8.5" x 14")
        (792, 1224),  # Tabloid (11" x 17")
        (672, 896),   # Base Nougat model size
        (1191, 1684), # A2
        (842, 1191),  # A3
        (595, 842),   # A4
        (420, 595),   # A5
        (298, 420),   # A6
        (729, 1032),  # B4
        (516, 729),   # B5
        (1032, 1456), # B3
        (672, 896),   # Base Nougat model size
        (841, 1189),  # C4 Envelope
        (624, 918),   # C5 Envelope
        (458, 648),   # C6 Envelope
        (709, 1001),  # RA4
        (607, 860),   # RA5
        (672, 896),   # Base Nougat model size
        (684, 1000),  # DL Envelope
        (850, 1100),  # ANSI A
        (672, 896),   # Base Nougat model size
    ]
    available_fonts = [
        "Rubik", "Cairo", "Almarai", "Tajawal", "IBM Plex Sans Arabic", "Amiri", "Changa",
        "El Messiri", "Noto Nashh Arabic", "Readex Pro", "Baloo Bhaijaan 2", "Mada",
        "Lalezar", "Reem Kufi", "Lemonada", "Alexandria", "Vazirmatn", "Lateef",
        "Jomhuria", "Aref Ruqaa", "Rakkas", "Mirza", "Ruwudu", "Katibeh", "Gulzar",
        "Marhey", "Noto Kufi Arabic", "Noto Sans Arabic",
    ]
    available_font_weights = [400, 500, 600, 700, 800, 900]
    available_font_sizes = [14, 16, 20, 24, 28, 32, 34, 36, 38, 40]
    dataset_output_path = Path(__file__).parent / "hindawi_dataset3"
    if dataset_output_path.exists():
        shutil.rmtree(dataset_output_path)
    dataset_output_path.mkdir(exist_ok=True)

    metadata_file = dataset_output_path / "metadata.csv"

    # Write CSV header
    with open(metadata_file, "w", newline='', encoding='utf-8') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["image_path", "markdown"])

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
    df = pd.DataFrame(ds)[8000:10000]
    print(f"Total records: {len(df)}")

    # Use multiprocessing to parallelize the processing with a progress bar
    with multiprocessing.Pool(48) as pool:
        list(tqdm(pool.imap(process_html, enumerate(df["html_content"])), total=len(df)))

    # Remove lock file
    remove_lock_file(metadata_file)
    
    print("Processing complete!")