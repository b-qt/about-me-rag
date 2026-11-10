""" Document: Image Converter Script
    
    This script finds all JPG/PNG images in the specified folder and converts them to PDF format then removes the original images. The converted PDFs are saved in the 'data/' directory.
"""

import os 
from PIL import Image
import pytesseract

DATA_DIR = "data"  # Directory to save converted images

def convert_images_to_pdf__init(folder_path: str) -> None:
    """
    Arguments:
        folder_path (str): The path to the folder containing images to convert.
    PURPOSE:
        - Go through the folder_path and find all images (JPEG, PNG, ..) and convert to PDF foramt.
        - Remove duplicates and save the converted images in the data/ folder.
    
    TODO: 
    """
    image_suffixes = (".jpg", ".jpeg", ".png", ".bmp", ".tiff") 
    
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(image_suffixes):
            image_path = os.path.join(folder_path, file_name)
            pdf_path = os.path.join(DATA_DIR, f"{os.path.splitext(file_name)[0]}.pdf")
            
            try:
                image = Image.open(image_path)
                
                # Convert image mode to RGB (required for saving as PDF)
                if image.mode != "RGB":
                    image = image.convert("RGB")
                    
                image.save(pdf_path, "PDF", resolution=100.0)
                print(f"Converted: {file_name} -> {os.path.basename(pdf_path)}")
                
                # Remove the original image after conversion
                os.remove(image_path)
                print(f"Removed original image: {file_name}")
            except Exception as e:
                print(f"Failed to convert {file_name}: {e}")    
       
def convert_images_to_txt__init(folder_path: str) -> None:
    """
    Arguments:
        folder_path (str): The path to the folder containing images to convert.
    PURPOSE:
        - Go through the folder_path and find all images (JPEG, PNG, ..) and convert to TXT foramt.
        - Remove duplicates and save the converted images in the data/ folder.         
    """
    image_suffixes = (".jpg", ".jpeg", ".png", ".bmp", ".tiff") 
    
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(image_suffixes):
            image_path = os.path.join(folder_path, file_name)
            txt_path = os.path.join(DATA_DIR, f"{os.path.splitext(file_name)[0]}.txt")
            
            try:
                # Use pytesseract to extract text from the image
                text = pytesseract.image_to_string(Image.open(image_path))
                
                # Save the extracted text to a .txt file
                with open(txt_path, 'w', encoding='utf-8') as txt_file:
                    txt_file.write(text)
                
                print(f"Converted: {file_name} -> {os.path.basename(txt_path)}")
                
                # Remove the original image after conversion
                #os.remove(image_path)
                #print(f"Removed original image: {file_name}")
            except Exception as e:
                print(f"Failed to convert {file_name}: {e}")
    

#-----------------------------------------------------------------#
""" Document: Image Converter Script (Optimized for OCR)
    
    This script finds all JPG/PNG/Screenshot images in the specified folder,
    applies image preprocessing (grayscale & contrast enhancement) for maximum OCR accuracy,
    extracts the text using pytesseract, and saves them as .txt files in the 'data/' directory.
"""

import os                                     # system operations
import re                                     # regex for text cleaning
from PIL import Image, ImageEnhance, ImageOps # image processing
import pytesseract                            # OCR engine

DATA_DIR = "data"                             # Directory to save text files

def preprocess_imageFor_ocr(image_path: str) -> Image.Image:
    """
    Applies image enhancements to maximize OCR accuracy:
    1. Converts image to grayscale.
    2. Enhances contrast to sharpen text edges.
    3. Resizes if the resolution is too low.
    """
    image = Image.open(image_path)
    
    # Convert to RGB if it has an alpha channel (RGBA)
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
        
    # Convert to Grayscale
    gray_image = ImageOps.grayscale(image)
    
    # Boost Contrast
    enhancer = ImageEnhance.Contrast(gray_image)
    enhanced_image = enhancer.enhance(2.0)  # Increase contrast factor
    
    return enhanced_image

def convert_images_to_txt__init_2(folder_path: str) -> None:
    """
    Arguments:
        folder_path (str): The path to the folder containing images to convert.
    PURPOSE:
        - Scan folder for images, preprocess them for clear OCR reading, 
          and save the extracted text into the data/ folder.
    """
    image_suffixes = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp") 
    
    if not os.path.exists(folder_path):
        print(f"Directory {folder_path} does not exist.")
        return

    files = os.listdir(folder_path)
    image_files = [f for f in files if f.lower().endswith(image_suffixes)]
    
    if not image_files:
        print(f"No valid image files found in {folder_path}.")
        return

    for file_name in image_files:
        image_path = os.path.join(folder_path, file_name)
        txt_path = os.path.join(DATA_DIR, f"{os.path.splitext(file_name)[0]}.txt")
        
        try:
            # Preprocess the image for better OCR results
            processed_image = preprocess_imageFor_ocr(image_path)
            
            # Use pytesseract with custom configuration for block text layout
            custom_config = r'--oem 3 --psm 6'
            text = pytesseract.image_to_string(processed_image, config=custom_config)
            
            # Save the extracted text to a .txt file
            with open(txt_path, 'w', encoding='utf-8') as txt_file:
                txt_file.write(text)
            
            print(f"Successfully converted: {file_name} -> {os.path.basename(txt_path)}")
            
            # Optional: Uncomment if you want to remove the source image after conversion
            # os.remove(image_path)
            # print(f"Removed original image: {file_name}")
            
        except Exception as e:
            print(f"Failed to convert {file_name}: {e}")

def convert_images_to_txt__init_3(folder_path: str) -> None:
    """
    Scans the folder for screenshots/images, upscales them 2x for optimal OCR DPI,
    and extracts clean text using Tesseract's English dictionary.
    """
    image_suffixes = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp") 
    
    if not os.path.exists(folder_path):
        print(f"Directory {folder_path} does not exist.")
        return

    files = os.listdir(folder_path)
    image_files = [f for f in files if f.lower().endswith(image_suffixes)]
    
    if not image_files:
        print(f"No valid image files found in {folder_path}.")
        return

    for file_name in image_files:
        image_path = os.path.join(folder_path, file_name)
        txt_path = os.path.join(DATA_DIR, f"{os.path.splitext(file_name)[0]}.txt")
        
        try:
            image = Image.open(image_path)
            
            # Convert RGBA/Palette images to RGB
            if image.mode in ("RGBA", "P"):
                image = image.convert("RGB")
            
            # Upscale resolution 2x using Bicubic interpolation. 
            # This turns low-DPI screen fonts into high-DPI text that Tesseract can read accurately.
            width, height = image.size
            upscaled_image = image.resize((width * 2, height * 2), Image.Resampling.BICUBIC)
            
            # Configure Tesseract:
            # - --oem 3: Default Neural net LSTM engine
            # - --psm 3: Fully automatic page segmentation
            # - -l eng: Explicitly enforce English dictionary to prevent language confusion
            custom_config = r'--oem 3 --psm 3 -l eng'
            text = pytesseract.image_to_string(upscaled_image, config=custom_config)
            
            # Save the clean extracted text to a .txt file
            with open(txt_path, 'w', encoding='utf-8') as txt_file:
                txt_file.write(text)
            
            print(f"Successfully converted: {file_name} -> {os.path.basename(txt_path)}")
            
        except Exception as e:
            print(f"Failed to convert {file_name}: {e}")
     

def clean_ocr_text(raw_text: str) -> str:
    """
    Post-processes raw OCR output to join wrapped lines into continuous sentences,
    while preserving new lines for bullet points (*, ©) and paragraph breaks.
    """
    lines = raw_text.splitlines()
    cleaned_lines = []
    current_paragraph = []
    
    # Define bullet point markers to keep on separate lines
    bullet_prefixes = ('*', '©', '-', '•')

    for line in lines:
        stripped = line.strip()
        
        if not stripped:
            # Empty line indicates a paragraph break
            if current_paragraph:
                cleaned_lines.append(" ".join(current_paragraph))
                current_paragraph = []
            cleaned_lines.append("")
        elif stripped.startswith(bullet_prefixes):
            # Flush any ongoing paragraph before the bullet point
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

def convert_images_to_txt(folder_path: str) -> None:
    """
    Scans the folder for screenshots/images, upscales them 2x for optimal OCR DPI,
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
            
            if image.mode in ("RGBA", "P"):
                image = image.convert("RGB")
            
            # Upscale 2x for high-DPI clarity
            width, height = image.size
            upscaled_image = image.resize((width * 2, height * 2), Image.Resampling.BICUBIC)
            
            # Extract text via Tesseract
            custom_config = r'--oem 3 --psm 3 -l eng'
            raw_text = pytesseract.image_to_string(upscaled_image, config=custom_config)
            
            # Apply our custom line-wrapping cleanup
            formatted_text = clean_ocr_text(raw_text)
            
            # Save to text file
            with open(txt_path, 'w', encoding='utf-8') as txt_file:
                txt_file.write(formatted_text)
            
            print(f"Successfully converted & formatted: {file_name} -> {os.path.basename(txt_path)}")
            
        except Exception as e:
            print(f"Failed to convert {file_name}: {e}")

if __name__ == "__main__":
    convert_images_to_txt(DATA_DIR)