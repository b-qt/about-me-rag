""" Document: Universal Image-to-Text Converter (Optimized for PDFs & Word Docs)
"""

import os 
import re
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import pytesseract

DATA_DIR = "data"  # Directory to save converted text files

def clean_ocr_text(raw_text: str) -> str:
    """
    Post-processes raw OCR output to join wrapped lines into continuous sentences,
    while preserving new lines for bullet points (*, ©) and paragraph breaks.
    """
    lines = raw_text.splitlines()
    cleaned_lines = []
    current_paragraph = []
    
    # Define bullet point markers to keep on separate lines
    bullet_prefixes = ('*', '©', '-', '•', 'o', '1.', '2.', '3.', '4.', '5.')

    for line in lines:
        stripped = line.strip()
        
        if not stripped:
            # Empty line indicates a paragraph break
            if current_paragraph:
                cleaned_lines.append(" ".join(current_paragraph))
                current_paragraph = []
            cleaned_lines.append("")
        elif stripped.startswith(bullet_prefixes) or len(stripped) < 3:
            # Flush any ongoing paragraph before headers or bullets
            if current_paragraph:
                cleaned_lines.append(" ".join(current_paragraph))
                current_paragraph = []
            cleaned_lines.append(stripped)
        else:
            # Accumulate normal text lines to join them into a continuous sentence
            current_paragraph.append(stripped)
            
    # Flush any remaining text in the buffer
    if current_paragraph:
        cleaned_lines.append(" ".join(current_paragraph))
        
    # Join everything back together, cleaning up duplicate consecutive newlines
    result = "\n".join(cleaned_lines)
    result = re.sub(r'\n{3,}', '\n\n', result)
    return result

def preprocess_word_screenshot(image: Image.Image) -> Image.Image:
    """
    Preprocesses screenshots specifically for Word / Document editors:
    - Crops out outer window frames/toolbars if present (focuses on page center).
    - Converts to high-contrast grayscale to eliminate editor window shadows.
    - Upscales resolution 2.5x for crisp character recognition.
    """
    # 1. Convert to RGB
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
        
    # 2. Convert to Grayscale
    gray = ImageOps.grayscale(image)
    
    # 3. Enhance contrast to cut through soft anti-aliasing in document UI
    enhancer = ImageEnhance.Contrast(gray)
    enhanced = enhancer.enhance(1.8)
    
    # 4. Upscale 2.5x for optimal OCR DPI resolution
    width, height = enhanced.size
    upscaled = enhanced.resize((int(width * 2.5), int(height * 2.5)), Image.Resampling.BICUBIC)
    
    return upscaled

def convert_images_to_txt(folder_path: str) -> None:
    """
    Scans the folder for screenshots/images, applies document-aware preprocessing,
    extracts text, and applies continuous line formatting.
    """
    image_suffixes = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp") 
    
    if not os.path.exists(folder_path):
        print(f"Directory {folder_path} does not exist.")
        return

    files = os.listdir(folder_path)
    image_files = [f for f in files if f.lower().endswith(image_suffixes)]
    
    if not image_files:
        print(f"No valid image files found in folder.")
        return

    for file_name in image_files:
        image_path = os.path.join(folder_path, file_name)
        txt_path = os.path.join(DATA_DIR, f"{os.path.splitext(file_name)[0]}.txt")
        
        try:
            image = Image.open(image_path)
            
            # Preprocess specifically for document layouts
            processed_image = preprocess_word_screenshot(image)
            
            # Configure Tesseract:
            # - --oem 3: Default LSTM OCR engine
            # - --psm 6: Assume a single uniform block of text (prevents Word margins/headers from scattering text)
            # - -l eng: Enforce English dictionary
            custom_config = r'--oem 3 --psm 6 -l eng'
            raw_text = pytesseract.image_to_string(processed_image, config=custom_config)
            
            # Apply our custom line-wrapping cleanup
            formatted_text = clean_ocr_text(raw_text)
            
            # Save to text file
            with open(txt_path, 'w', encoding='utf-8') as txt_file:
                txt_file.write(formatted_text)
            
            print(f"Successfully converted Word/PDF screenshot: {file_name} -> {os.path.basename(txt_path)}")
            
        except Exception as e:
            print(f"Failed to convert {file_name}: {e}")

if __name__ == "__main__":
    convert_images_to_txt(DATA_DIR)