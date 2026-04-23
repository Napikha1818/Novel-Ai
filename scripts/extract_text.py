"""
extract_text.py
===============
Ekstrak teks dari file EPUB dan PDF di folder data/raw/
Hasil disimpan di data/extracted/ sebagai file .txt per novel.

v2 — Perbaikan:
  - Skip halaman copyright, penerbit, ISBN
  - Deteksi dan skip teks rusak (tanpa spasi)
  - Pembersihan artefak lebih agresif
"""

import os
import sys
import re
from pathlib import Path

# EPUB
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup

# PDF
import fitz  # PyMuPDF


RAW_DIR = Path("data/raw")
EXTRACTED_DIR = Path("data/extracted")

# Kata kunci halaman yang harus di-skip
SKIP_KEYWORDS = [
    "sanksi pelanggaran",
    "undang-undang",
    "hak cipta",
    "isbn",
    "penerbit",
    "cetakan pertama",
    "cetakan kedua",
    "cetakan ketiga",
    "diterbitkan pertama",
    "gramedia pustaka",
    "kompas gramedia",
    "dilarang memperbanyak",
    "desain sampul",
    "foto sampul",
    "daftar isi",
    "prakata",
    "kata pengantar penerbit",
]


def is_junk_page(text: str) -> bool:
    """Deteksi apakah halaman ini sampah (copyright, penerbit, dll)."""
    text_lower = text.lower()

    # Cek kata kunci skip
    matches = sum(1 for kw in SKIP_KEYWORDS if kw in text_lower)
    if matches >= 2:
        return True

    # Deteksi teks rusak: kata-kata terlalu panjang (tanpa spasi)
    words = text.split()
    if not words:
        return True

    long_words = [w for w in words if len(w) > 40]
    if len(long_words) > len(words) * 0.3:  # >30% kata terlalu panjang
        return True

    # Halaman terlalu pendek (kemungkinan header/footer saja)
    if len(words) < 30:
        return True

    return False


def clean_text(text: str) -> str:
    """Bersihkan teks dari artefak konversi PDF/EPUB."""

    # === SOFT HYPHEN ===
    # Hapus soft hyphen (U+00AD) dan gabungkan kata yang terpotong
    # "ke\xadluarkan" → "keluarkan"
    text = text.replace("\u00ad", "")

    # === HARD HYPHEN DI AKHIR BARIS ===
    # "meng-\nambil" → "mengambil"
    text = re.sub(r"-\n(\w)", r"\1", text)

    # === SPASI NYASAR DI TENGAH KATA ===
    # Pola: huruf kecil + spasi + huruf kecil di tengah kata
    # "pem bantai" → "pembantai", "fan tastis" → "fantastis"
    # Hati-hati: hanya perbaiki jika potongannya pendek (1-4 huruf)
    text = re.sub(r"(\w{2,}) ([a-z]{1,4})(?=\s|[.,;:!?\n])", _fix_broken_word, text)

    # === HEADER/FOOTER BERULANG ===
    text = re.sub(r"(?i)cantik\s*itu\s*luka\s*\n", "\n", text)
    text = re.sub(r"(?i)eka\s*kurniawan\s*\n", "\n", text)
    text = re.sub(r"(?i)gramedia\s*pustaka\s*utama\s*\n", "\n", text)
    text = re.sub(r"(?i)penerbit\s*pt\s*gramedia\s*\n", "\n", text)

    # === NOMOR HALAMAN ===
    text = re.sub(r"^\d{1,4}$", "", text, flags=re.MULTILINE)

    # === BARIS SIMBOL SAJA ===
    text = re.sub(r"^[^\w\s]{1,5}$", "", text, flags=re.MULTILINE)

    # === SPASI HILANG ANTAR KATA UMUM ===
    # Perbaiki kata-kata yang menempel karena PDF kehilangan spasi
    # Pola: kata + partikel/kata ganti yang menempel
    _glued_patterns = [
        # Kata ganti yang menempel: "ketikaia" → "ketika ia"
        (r"(\w{3,})(ia|itu|ini|aku|kau|dia|kami|kita|mereka)(?=\s|[.,;:!?\n])", r"\1 \2"),
        # Preposisi menempel di belakang: "berlarike" → "berlari ke"
        (r"(\w{3,})(ke|di|se|dan|atau|yang|dari|pada|untuk|dengan|dalam|oleh)(?=\s|[.,;:!?\n])", r"\1 \2"),
        # Preposisi menempel di depan: "kepadanya" sudah benar, tapi "kembalipe" → salah
        # Partikel menempel: "bagaimanapunia" → "bagaimanapun ia"
        (r"(pun|lah|kah|tah)(ia|itu|ini|aku|dia|mereka|kami|kita)", r"\1 \2"),
    ]
    for pattern, replacement in _glued_patterns:
        text = re.sub(pattern, replacement, text)

    # === SPASI & NEWLINE BERLEBIHAN ===
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# Kamus sederhana untuk validasi gabungan kata
# Prefix umum bahasa Indonesia yang sering terpotong di PDF
_PREFIXES = {
    "mem", "men", "meng", "meny", "me", "ber", "per", "ter", "di",
    "ke", "se", "pem", "pen", "peng", "peny", "pe", "pel",
}

# Suffix umum yang sering terpotong
_SUFFIXES = {
    "kan", "an", "nya", "lah", "kah", "pun", "mu", "ku",
    "i", "wan", "wati", "man", "is", "if", "al",
}


def _fix_broken_word(match):
    """
    Perbaiki kata yang terpotong oleh spasi nyasar.
    Hanya gabungkan jika bagian pertama adalah prefix ATAU bagian kedua adalah suffix.
    """
    part1 = match.group(1)
    part2 = match.group(2)

    # Cek apakah part1 berakhir dengan prefix umum
    for prefix in _PREFIXES:
        if part1.lower().endswith(prefix) or part1.lower() == prefix:
            return part1 + part2

    # Cek apakah part2 adalah suffix umum
    if part2.lower() in _SUFFIXES:
        return part1 + part2

    # Cek apakah part2 sangat pendek (1-2 huruf) — kemungkinan besar terpotong
    if len(part2) <= 2 and part2.isalpha():
        return part1 + part2

    # Jika tidak yakin, biarkan apa adanya
    return match.group(0)


def extract_epub(filepath: Path) -> str:
    """Ekstrak teks dari file EPUB."""
    book = epub.read_epub(str(filepath), options={"ignore_ncx": True})
    chapters = []

    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_content(), "html.parser")
            text = soup.get_text(separator="\n")
            text = clean_text(text)
            if len(text) > 100 and not is_junk_page(text):
                chapters.append(text)

    return "\n\n---BAB_SEPARATOR---\n\n".join(chapters)


def extract_pdf(filepath: Path) -> str:
    """Ekstrak teks dari file PDF."""
    doc = fitz.open(str(filepath))
    pages = []
    skipped = 0

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()
        text = clean_text(text)

        if is_junk_page(text):
            skipped += 1
            continue

        pages.append(text)

    doc.close()

    if skipped > 0:
        print(f"   🗑️  {skipped} halaman sampah di-skip (copyright, penerbit, dll)")

    return "\n\n".join(pages)


def main():
    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)

    epub_files = list(RAW_DIR.glob("*.epub"))
    # Deduplikasi PDF (hindari .pdf dan .PDF double)
    pdf_set = set()
    for p in RAW_DIR.glob("*.pdf"):
        pdf_set.add(p)
    for p in RAW_DIR.glob("*.PDF"):
        pdf_set.add(p)
    pdf_files = sorted(pdf_set)

    if not epub_files and not pdf_files:
        print("❌ Tidak ada file EPUB atau PDF di data/raw/")
        print("   Taruh file novel di folder data/raw/ lalu jalankan ulang.")
        sys.exit(1)

    print(f"📚 Ditemukan {len(epub_files)} EPUB, {len(pdf_files)} PDF\n")

    for filepath in epub_files:
        print(f"📖 Mengekstrak EPUB: {filepath.name}")
        text = extract_epub(filepath)
        output_name = filepath.stem + ".txt"
        output_path = EXTRACTED_DIR / output_name
        output_path.write_text(text, encoding="utf-8")
        word_count = len(text.split())
        print(f"   ✅ {word_count:,} kata → {output_path}\n")

    for filepath in pdf_files:
        print(f"📖 Mengekstrak PDF: {filepath.name}")
        text = extract_pdf(filepath)
        output_name = filepath.stem + ".txt"
        output_path = EXTRACTED_DIR / output_name
        output_path.write_text(text, encoding="utf-8")
        word_count = len(text.split())
        print(f"   ✅ {word_count:,} kata → {output_path}\n")

    # Ringkasan total
    total_words = 0
    for txt in EXTRACTED_DIR.glob("*.txt"):
        content = txt.read_text(encoding="utf-8")
        total_words += len(content.split())

    print(f"🎉 Selesai! Total: {total_words:,} kata di data/extracted/")


if __name__ == "__main__":
    main()
