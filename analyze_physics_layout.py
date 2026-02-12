import fitz
import config
import os

# Ensure image directory exists
if not os.path.exists(config.IMAGE_DIR):
    os.makedirs(config.IMAGE_DIR)

def verify_layout_detection(page_num=0):
    try:
        doc = fitz.open(config.PDF_PATH)
    except Exception as e:
        print(f"Error: Could not find {config.PDF_PATH}. Check your config.py.")
        return

    page = doc[page_num]
    # "dict" format is essential for multimodal work as it identifies images/drawings
    blocks = page.get_text("dict")["blocks"]
    
    print(f"--- Layout Verification: {config.PDF_PATH} | Page {page_num + 1} ---")
    print(f"Total blocks detected: {len(blocks)}\n")

    for i, b in enumerate(blocks):
        bbox = b["bbox"]
        # Identify if block is Text (0) or Image (1)
        b_type = "TEXT" if "lines" in b else "IMAGE/GRAPHIC"
        
        print(f"Block {i} | Type: {b_type} | BBox: ({bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f})")
        
        if b_type == "TEXT":
            # Extract snippet
            text_snippet = "".join([span["text"] for line in b["lines"] for span in line["spans"]])
            print(f"   Content: {text_snippet[:60].strip()}...")
        else:
            print(f"   [Multimodal Zone detected - Will require cropping/OCR later]")
        
        print("-" * 30)
        
        # Limit output for first check
        if i >= 15: 
            print("... and more blocks. Check if these look correct.")
            break

    doc.close()

if __name__ == "__main__":
    verify_layout_detection()