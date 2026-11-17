"""OCR screenshots or photos of documents into text files (offline helper, not used by the app).

Merges the earlier image_converter.py / image_converter_2.py variants into one script:
2x bicubic upscaling, optional grayscale + contrast for document screenshots, Tesseract
LSTM engine, then joining wrapped lines while keeping bullets and paragraph breaks.

Raw OCR output is noisy, so it goes to private/ocr/ by default. Read it, fix it, drop anything private (salaries, contact details), and save the clean version in knowledge/.

    pip install -r requirements-dev.txt     # plus the tesseract-ocr system package
    python scripts/ocr_images.py private/screenshots
    python scripts/ocr_images.py private/screenshots --mode document --out private/ocr
"""
from __future__ import annotations #-- 

import argparse
import re
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
BULLETS = ("*", "©", "-", "•", "●", "○", "e ", "+ ")
ROOT = Path(__file__).resolve().parents[1]


def clean_ocr_text(raw: str) -> str:
    """
        Clean up the raw OCR text by joining wrapped lines and keeping bullets and paragraph breaks.
    """
    lines, paragraph = [], []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            if paragraph:
                lines.append(" ".join(paragraph))
                paragraph = []
            lines.append("")
        elif stripped.startswith(BULLETS) or len(stripped) < 3:
            if paragraph:
                lines.append(" ".join(paragraph))
                paragraph = []
            lines.append(stripped)
        else:
            paragraph.append(stripped)
    if paragraph:
        lines.append(" ".join(paragraph))
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip() + "\n"


def preprocess(image, mode: str):
    """ 
    Preprocess the image for OCR: convert to RGB, optionally grayscale + contrast, and upscale.
    """
    from PIL import Image, ImageEnhance, ImageOps

    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
    scale = 2.0
    if mode == "document":
        image = ImageEnhance.Contrast(ImageOps.grayscale(image)).enhance(1.8)
        scale = 2.5
    return image.resize((int(image.width * scale), int(image.height * scale)), Image.Resampling.BICUBIC)


def convert(folder: Path, out: Path, mode: str, overwrite: bool) -> None:
    """ Convert images in a folder to text files using OCR. """
    import pytesseract
    from PIL import Image

    out.mkdir(parents=True, exist_ok=True)
    psm = 6 if mode == "document" else 3
    images = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        print(f"No images found in {folder}")
    for path in images:
        target = out / f"{path.stem}.txt"
        if target.exists() and not overwrite:
            print(f"skip  {path.name} (already converted; use --overwrite)")
            continue
        text = pytesseract.image_to_string(preprocess(Image.open(path), mode), config=f"--oem 3 --psm {psm} -l eng")
        target.write_text(clean_ocr_text(text), encoding="utf-8")
        print(f"ok    {path.name} -> {target.relative_to(ROOT) if target.is_relative_to(ROOT) else target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "private" / "ocr")
    parser.add_argument("--mode", choices=["screen", "document"], default="screen",
                        help="screen: web pages and job ads; document: Word/PDF screenshots")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    convert(args.folder, args.out, args.mode, args.overwrite)
