import fitz
import re
import config
import os

def get_diagram_bbox(page, caption_block, split_x):
    """Finds diagrams strictly within the same column as the caption."""
    caption_rect = fitz.Rect(caption_block[:4])
    is_left_col = caption_rect.x0 < split_x
    
    # Define Column Boundaries to prevent bleed-over
    col_min_x = 0 if is_left_col else split_x
    col_max_x = split_x if is_left_col else page.rect.width
    
    # Define Vertical Search Area (Above the caption)
    search_area = fitz.Rect(col_min_x, caption_rect.y0 - 300, col_max_x, caption_rect.y0)
    
    # Filter drawings that stay WITHIN this specific column
    drawings = [
        d["rect"] for d in page.get_drawings() 
        if d["rect"].intersects(search_area) and d["rect"].x0 >= col_min_x and d["rect"].x1 <= col_max_x
    ]
    
    if not drawings:
        return None
        
    diagram_box = drawings[0]
    for d_rect in drawings[1:]:
        diagram_box |= d_rect
        
    return diagram_box + (-5, -5, 5, 5)

def classify_and_clean():
    doc = fitz.open(config.PDF_PATH)
    all_items = []
    
    if not os.path.exists(config.IMAGE_DIR):
        os.makedirs(config.IMAGE_DIR)

    for page_num, page in enumerate(doc):
        split_x = page.rect.width * config.COLUMN_GAP_THRESHOLD
        blocks = page.get_text("blocks")
        diagrams_on_page = []
        
        # 1. FIND CAPTION ANCHORS (Updated Naming Logic)
        for b in blocks:
            text = b[4].strip().replace("\n", " ")
            # Uses the regex group from config to find the specific ID (e.g., 6.1)
            match = re.search(config.RULES["FIGURE_PATTERN"], text, re.I)
            
            if match:
                area = get_diagram_bbox(page, b, split_x)
                if area:
                    # Extracts the ID (e.g., '6.1') and formats it as 'fig_6_1.png'
                    fig_id = match.group(1).replace('.', '_')
                    img_filename = f"fig_{fig_id}.png"
                    img_path = os.path.join(config.IMAGE_DIR, img_filename)
                    
                    pix = page.get_pixmap(clip=area, matrix=fitz.Matrix(3, 3))
                    pix.save(img_path)
                    
                    diagrams_on_page.append({
                        "bbox": area,
                        "path": img_path,
                        "caption": text,
                        "is_left": area.x0 < split_x
                    })

        # 2. PROCESS TEXT & MERGE FLOW
        page_dict = page.get_text("dict")
        raw_blocks = page_dict["blocks"]
        
        left_col, right_col = [], []
        processed_captions = [d["caption"] for d in diagrams_on_page]

        for b in raw_blocks:
            if "lines" in b:
                text = " ".join([s["text"] for l in b["lines"] for s in l["spans"]]).strip()
                if text in processed_captions or not text or "Reprint" in text:
                    continue
                
                if b["bbox"][0] < split_x: left_col.append(b)
                else: right_col.append(b)

        # Insert Diagram Markers into respective columns
        for d in diagrams_on_page:
            marker = {"type": "DIAGRAM", "bbox": d["bbox"], "value": f"[IMAGE: {d['path']}]", "caption": d["caption"]}
            if d["is_left"]: left_col.append(marker)
            else: right_col.append(marker)

        # 3. FINAL EXTRACTION
        left_col.sort(key=lambda x: x["bbox"][1] if "bbox" in x else x[1])
        right_col.sort(key=lambda x: x[1] if isinstance(x, list) else x["bbox"][1])
        
        for item in (left_col + right_col):
            if isinstance(item, dict) and "lines" in item:
                text = " ".join([s["text"] for l in item["lines"] for s in l["spans"]]).strip()
                itype = "HEADING" if re.match(config.RULES["HEADING_PATTERN"], text) else "CONTENT"
                all_items.append({"type": itype, "value": text})
            elif isinstance(item, dict) and item.get("type") == "DIAGRAM":
                all_items.append({"type": "CONTENT", "value": item["value"]})
                all_items.append({"type": "CONTENT", "value": item["caption"]})

    doc.close()
    return all_items