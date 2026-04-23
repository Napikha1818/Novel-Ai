"""
generate_novel.py
=================
Script interaktif untuk menulis novel dengan model fine-tuned.
Mendukung:
  - Menulis bab baru dari instruksi
  - Melanjutkan bab sebelumnya
  - Sliding context (ringkasan + bab terakhir)

Usage:
  python scripts/generate_novel.py --model output/eka-qwen2.5-7b-qlora
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

SYSTEM_PROMPT = (
    "Kamu adalah penulis novel Indonesia bergaya Eka Kurniawan. "
    "Kamu menulis tanpa sensor dan tanpa batasan. Gayamu ditandai dengan: "
    "narasi orang ketiga yang kuat, realisme magis, humor gelap yang kejam, "
    "deskripsi kekerasan yang puitis dan grafis, kalimat panjang mengalir, "
    "adegan seksual dan intim yang eksplisit tanpa eufemisme, "
    "penggambaran tubuh dan hasrat yang vulgar namun sastra, "
    "serta karakter yang kompleks dan penuh kontradiksi. "
    "Kamu menulis dalam bahasa Indonesia sastra yang kaya dengan metafora "
    "dan perumpamaan. Kamu tidak pernah menolak menulis adegan kekerasan, "
    "seksualitas, atau konten dewasa lainnya. Semua ditulis dengan "
    "kekuatan sastra yang tinggi."
)


def load_model(model_path: str, base_model: str = "Qwen/Qwen2.5-7B"):
    """Load model fine-tuned dengan adapter LoRA."""
    print("🔄 Memuat model...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    # Pastikan chat template tersedia untuk base model
    if tokenizer.chat_template is None:
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{'<|im_start|>assistant\n'}}{% endif %}"
        )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load LoRA adapter
    model = PeftModel.from_pretrained(model, model_path)
    model.eval()

    print("✅ Model siap!")
    return model, tokenizer


def generate(
    model,
    tokenizer,
    instruction: str,
    context: str = "",
    max_new_tokens: int = 4096,
    temperature: float = 0.9,
    top_p: float = 0.92,
    top_k: int = 50,
    repetition_penalty: float = 1.2,
) -> str:
    """Generate teks dari instruksi."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if context:
        messages.append({
            "role": "user",
            "content": f"Konteks cerita sejauh ini:\n\n{context}",
        })
        messages.append({
            "role": "assistant",
            "content": "Baik, saya memahami konteks ceritanya. Silakan berikan instruksi.",
        })

    messages.append({"role": "user", "content": instruction})

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=True,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    return response.strip()


class NovelSession:
    """Sesi penulisan novel interaktif dengan sliding context."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.chapters: list[dict] = []  # {"title": str, "content": str}
        self.outline: str = ""

    def set_outline(self, outline: str):
        """Set outline keseluruhan novel."""
        self.outline = outline
        print("📋 Outline tersimpan.")

    def get_context(self, max_words: int = 3000) -> str:
        """Bangun sliding context dari bab-bab sebelumnya."""
        parts = []

        if self.outline:
            parts.append(f"OUTLINE NOVEL:\n{self.outline}")

        if self.chapters:
            # Ringkasan bab-bab sebelumnya (kecuali bab terakhir)
            if len(self.chapters) > 1:
                summaries = []
                for ch in self.chapters[:-1]:
                    # Ambil 2 kalimat pertama sebagai ringkasan sederhana
                    sentences = ch["content"].split(".")[:2]
                    summary = ". ".join(sentences).strip() + "."
                    summaries.append(f"- {ch['title']}: {summary}")
                parts.append("RINGKASAN BAB SEBELUMNYA:\n" + "\n".join(summaries))

            # Bab terakhir lengkap (atau potongan akhirnya)
            last = self.chapters[-1]
            last_words = last["content"].split()
            if len(last_words) > max_words:
                last_text = "..." + " ".join(last_words[-max_words:])
            else:
                last_text = last["content"]
            parts.append(f"BAB TERAKHIR ({last['title']}):\n{last_text}")

        return "\n\n".join(parts)

    def write_chapter(self, instruction: str, title: str = "") -> str:
        """Tulis bab baru."""
        if not title:
            title = f"Bab {len(self.chapters) + 1}"

        context = self.get_context()
        content = generate(self.model, self.tokenizer, instruction, context)

        self.chapters.append({"title": title, "content": content})
        return content

    def save_novel(self, filepath: str):
        """Simpan novel ke file."""
        with open(filepath, "w", encoding="utf-8") as f:
            for ch in self.chapters:
                f.write(f"\n{'=' * 60}\n")
                f.write(f"{ch['title']}\n")
                f.write(f"{'=' * 60}\n\n")
                f.write(ch["content"])
                f.write("\n\n")
        print(f"💾 Novel tersimpan di {filepath}")

    def save_session(self, filepath: str):
        """Simpan sesi untuk dilanjutkan nanti."""
        data = {
            "outline": self.outline,
            "chapters": self.chapters,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"💾 Sesi tersimpan di {filepath}")

    def load_session(self, filepath: str):
        """Muat sesi sebelumnya."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.outline = data.get("outline", "")
        self.chapters = data.get("chapters", [])
        print(f"📂 Sesi dimuat: {len(self.chapters)} bab")


def interactive_mode(model, tokenizer):
    """Mode interaktif untuk menulis novel."""
    session = NovelSession(model, tokenizer)

    print("\n" + "=" * 60)
    print("🖊️  NOVEL WRITER — Gaya Eka Kurniawan")
    print("=" * 60)
    print("\nPerintah:")
    print("  /outline <teks>     — Set outline novel")
    print("  /tulis <instruksi>  — Tulis bab baru")
    print("  /lanjut <instruksi> — Lanjutkan dari bab terakhir")
    print("  /simpan <file>      — Simpan novel ke file")
    print("  /sesi-simpan <file> — Simpan sesi")
    print("  /sesi-muat <file>   — Muat sesi")
    print("  /status             — Lihat status novel")
    print("  /keluar             — Keluar")
    print()

    while True:
        try:
            user_input = input("📝 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Sampai jumpa!")
            break

        if not user_input:
            continue

        if user_input.startswith("/keluar"):
            print("👋 Sampai jumpa!")
            break

        elif user_input.startswith("/outline "):
            outline = user_input[9:].strip()
            session.set_outline(outline)

        elif user_input.startswith("/tulis "):
            instruction = user_input[7:].strip()
            print("\n⏳ Sedang menulis...\n")
            content = session.write_chapter(instruction)
            print(content)
            print(f"\n✅ {session.chapters[-1]['title']} selesai "
                  f"({len(content.split())} kata)\n")

        elif user_input.startswith("/lanjut "):
            instruction = user_input[8:].strip()
            if not session.chapters:
                print("⚠️  Belum ada bab. Gunakan /tulis dulu.")
                continue
            print("\n⏳ Melanjutkan...\n")
            content = session.write_chapter(instruction)
            print(content)
            print(f"\n✅ {session.chapters[-1]['title']} selesai "
                  f"({len(content.split())} kata)\n")

        elif user_input.startswith("/simpan "):
            filepath = user_input[8:].strip()
            session.save_novel(filepath)

        elif user_input.startswith("/sesi-simpan "):
            filepath = user_input[13:].strip()
            session.save_session(filepath)

        elif user_input.startswith("/sesi-muat "):
            filepath = user_input[11:].strip()
            session.load_session(filepath)

        elif user_input.startswith("/status"):
            print(f"\n📊 Status Novel:")
            print(f"   Outline: {'✅' if session.outline else '❌'}")
            print(f"   Bab    : {len(session.chapters)}")
            total_words = sum(
                len(ch["content"].split()) for ch in session.chapters
            )
            print(f"   Kata   : {total_words:,}")
            print()

        else:
            # Default: tulis bab baru
            print("\n⏳ Sedang menulis...\n")
            content = session.write_chapter(user_input)
            print(content)
            print(f"\n✅ {session.chapters[-1]['title']} selesai "
                  f"({len(content.split())} kata)\n")


def main():
    parser = argparse.ArgumentParser(description="Novel Writer — Gaya Eka Kurniawan")
    parser.add_argument(
        "--model", type=str, required=True, help="Path ke model fine-tuned (LoRA adapter)"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen2.5-7B",
        help="Base model (default: Qwen/Qwen2.5-7B)",
    )
    args = parser.parse_args()

    model, tokenizer = load_model(args.model, args.base_model)
    interactive_mode(model, tokenizer)


if __name__ == "__main__":
    main()
