"""
ocr_epub.py
===========
OCR untuk EPUB yang berisi gambar scan (bukan teks).
Khusus untuk: Perempuan_patah_hati.epub
"""

import io
import sys
from pathlib import Path

import ebooklib
from ebooklib import epub
from PIL import Image
import easyocr

RAW_DIR = Path("data/raw")
EXTRACTED_DIR = Path("data/extracted")


def main():
    epub_path = RAW_DIR / "Perempuan_patah_hati.epub"

    if not epub_path.exists():
        print(f"❌ File tidak ditemukan: {epub_path}")
        sys.exit(1)

    print("🔄 Memuat EasyOCR (bahasa Indonesia)...")
    reader = easyocr.Reader(["id", "en"], gpu=True)

    print(f"📖 Membuka: {epub_path.name}")
    book = epub.read_epub(str(epub_path), options={"ignore_ncx": True})

    # Kumpulkan semua gambar PNG (halaman scan)
    images = []
    for item in book.get_items():
        name = item.get_name()
        if name.endswith(".png") and "index-" in name:
            # Ekstrak nomor halaman dari nama file
            try:
                page_num = int(name.split("index-")[1].split("_")[0])
            except (ValueError, IndexError):
                page_num = 999
            images.append((page_num, name, item.get_content()))

    # Sort berdasarkan nomor halaman
    images.sort(key=lambda x: x[0])
    print(f"   📄 {len(images)} halaman ditemukan\n")

    all_text = []
    for i, (page_num, name, content) in enumerate(images):
        print(f"   🔍 OCR halaman {page_num} ({i+1}/{len(images)})...", end=" ")
        try:
            import numpy as np
            img = Image.open(io.BytesIO(content)).convert("RGB")
            img_array = np.array(img)
            results = reader.readtext(
                img_array,
                detail=0,
                paragraph=True,
            )
            page_text = "\n".join(results)
            word_count = len(page_text.split())
            print(f"{word_count} kata")

            if word_count > 5:  # Skip halaman kosong/gambar
                all_text.append(page_text)
        except Exception as e:
            print(f"ERROR: {e}")

    # Gabungkan dan simpan
    full_text = "\n\n".join(all_text)
    output_path = EXTRACTED_DIR / "Perempuan_patah_hati.txt"
    output_path.write_text(full_text, encoding="utf-8")

    total_words = len(full_text.split())
    print(f"\n✅ Selesai! {total_words:,} kata → {output_path}")


if __name__ == "__main__":
    main()
