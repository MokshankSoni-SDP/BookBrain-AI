import fitz
import config  # Import the central config

def extract_page_blocks(page_num=1):
    # Use the path defined in config.py
    doc = fitz.open(config.PDF_PATH)
    page = doc[page_num] 
    
    blocks = page.get_text("blocks")
    
    print(f"--- Block Extraction for {config.PDF_PATH} | Page {page_num + 1} ---")
    for b in blocks[:10]:
        x0, y0, x1, y1, content, block_no, block_type = b
        clean_content = content.replace("\n", " ").strip()[:50]
        
        print(f"Block {block_no} | Rect: ({x0:.1f}, {y0:.1f}, {x1:.1f}, {y1:.1f})")
        print(f"   Content: {clean_content}...")
        print("-" * 20)

    doc.close()

if __name__ == "__main__":
    extract_page_blocks()