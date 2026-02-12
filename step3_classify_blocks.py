import fitz
import re
import config
import os

def get_diagram_bbox(page, caption_block):
    """
    Given a caption block, looks upward to find the 
    associated diagram area by checking for drawings/images.
    """
    caption_rect = fitz.Rect(caption_block[:4])
    # Define a search area above the caption (roughly 250 points up)
    search_area = fitz.Rect(caption_rect.x0, caption_rect.y0 - 250, 
                            caption_rect.x1 + 50, caption_rect.y0)
    
    # Get all drawings in that search area
    drawings = [d["rect"] for d in page.get_drawings() if d["rect"].intersects(search_area)]
    
    if not drawings:
        return None
        
    # Merge all drawings into one large diagram box
    diagram_box = drawings[0]
    for d_rect in drawings[1:]:
        diagram_box |= d_rect
        
    # Add a small padding to catch labels like theta or P1
    return diagram_box + (-10, -10, 10, 10)

def classify_and_clean():
    doc = fitz.open(config.PDF_PATH)
    all_items = []
    img_counter = 0
    
    for page_num, page in enumerate(doc):
        page_width = page.rect.width
        split_x = page_width * config.COLUMN_GAP_THRESHOLD
        
        # Get blocks to find captions first
        blocks = page.get_text("blocks")
        diagrams_on_page = []
        
        # 1. FIND CAPTION ANCHORS
        for b in blocks:
            if re.match(config.RULES["FIGURE_PATTERN"], b[4].strip(), re.I):
                area = get_diagram_bbox(page, b)
                if area:
                    img_counter += 1
                    img_path = os.path.join(config.IMAGE_DIR, f"page_{page_num+1}_fig_{img_counter}.png")
                    
                    # Save snapshot
                    pix = page.get_pixmap(clip=area, matrix=fitz.Matrix(3, 3))
                    pix.save(img_path)
                    
                    diagrams_on_page.append({
                        "bbox": area,
                        "path": img_path,
                        "caption": b[4].strip()
                    })

        # 2. PROCESS TEXT FLOW
        page_dict = page.get_text("dict")
        raw_blocks = page_dict["blocks"]
        
        left_col, right_col = [], []
        processed_captions = [d["caption"] for d in diagrams_on_page]

        for b in raw_blocks:
            if "lines" in b:
                text = " ".join([s["text"] for l in b["lines"] for s in l["spans"]]).strip()
                
                # Skip if this text is a caption we already handled for a diagram
                if text in processed_captions or not text or "Reprint" in text:
                    continue
                
                # Partition
                if b["bbox"][0] < split_x: left_col.append(b)
                else: right_col.append(b)

        # 3. MERGE DIAGRAMS INTO FLOW
        for d in diagrams_on_page:
            marker = {"type": "DIAGRAM", "bbox": d["bbox"], "value": f"[IMAGE: {d['path']}]", "caption": d["caption"]}
            if d["bbox"][0] < split_x: left_col.append(marker)
            else: right_col.append(marker)

        # Sort and Extract
        left_col.sort(key=lambda x: x["bbox"][1] if "bbox" in x else x[1])
        right_col.sort(key=lambda x: x["bbox"][1] if "bbox" in x else x[1])
        
        for item in (left_col + right_col):
            if "lines" in item:
                text = " ".join([s["text"] for l in item["lines"] for s in l["spans"]]).strip()
                itype = "HEADING" if re.match(config.RULES["HEADING_PATTERN"], text) else "CONTENT"
                all_items.append({"type": itype, "value": text})
            else:
                all_items.append({"type": "CONTENT", "value": item["value"]})
                all_items.append({"type": "CONTENT", "value": item["caption"]})

    doc.close()
    return all_items