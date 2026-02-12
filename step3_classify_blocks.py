import fitz
import re
import config
import os

# Ensure the image directory exists
if not os.path.exists(config.IMAGE_DIR):
    os.makedirs(config.IMAGE_DIR)

def get_smart_diagram_areas(page):
    """Detects areas containing vector drawings and labels."""
    paths = page.get_drawings()
    diagram_bboxes = []
    
    for p in paths:
        rect = p["rect"]
        if rect.width < 5 or rect.height < 5: continue # Ignore tiny artifacts
        
        found = False
        for i, bbox in enumerate(diagram_bboxes):
            # If a drawing is within 50 points of another, merge them into one diagram
            if rect.intersects(bbox + (-50, -50, 50, 50)):
                diagram_bboxes[i] = bbox | rect
                found = True
                break
        if not found:
            diagram_bboxes.append(rect)
    return diagram_bboxes

def classify_and_clean():
    doc = fitz.open(config.PDF_PATH)
    all_items = []
    fig_counter = 0
    
    for page_num, page in enumerate(doc):
        page_width = page.rect.width
        split_x = page_width * config.COLUMN_GAP_THRESHOLD
        
        # 1. Identify Diagram Areas first to avoid double-processing
        diagram_areas = get_smart_diagram_areas(page)
        
        # 2. Get all page blocks
        page_dict = page.get_text("dict")
        blocks = page_dict["blocks"]
        
        # Partition blocks into Left and Right columns
        left_col, right_col = [], []
        for b in blocks:
            # Skip any block that is inside a detected diagram area
            # We will handle diagrams as a single unit later
            if any(fitz.Rect(b["bbox"]).intersects(area) for area in diagram_areas):
                continue
            
            if b["bbox"][0] < split_x:
                left_col.append(b)
            else:
                right_col.append(b)

        # Sort columns vertically
        left_col.sort(key=lambda x: x["bbox"][1])
        right_col.sort(key=lambda x: x["bbox"][1])
        
        # Add diagrams to their respective columns based on their position
        for area in diagram_areas:
            fig_counter += 1
            img_filename = f"page_{page_num+1}_fig_{fig_counter}.png"
            img_path = os.path.join(config.IMAGE_DIR, img_filename)
            
            # Save the diagram with a margin to catch labels (P1, theta, etc.)
            snapshot_area = area + (-15, -15, 15, 15)
            pix = page.get_pixmap(clip=snapshot_area, matrix=fitz.Matrix(3, 3))
            pix.save(img_path)
            
            diagram_marker = {
                "type": "DIAGRAM",
                "bbox": area,
                "path": img_path,
                "value": f"[IMAGE: {img_path}]"
            }
            
            # Insert diagram into the correct column flow
            if area.x0 < split_x:
                left_col.append(diagram_marker)
            else:
                right_col.append(diagram_marker)

        # Final sort for each column to ensure diagrams are in the right sequence with text
        left_col.sort(key=lambda x: x["bbox"][1] if isinstance(x, dict) and "bbox" in x else x[1])
        right_col.sort(key=lambda x: x["bbox"][1] if isinstance(x, dict) and "bbox" in x else x[1])
        
        ordered_blocks = left_col + right_col

        # 3. Process ordered items into the final list
        i = 0
        while i < len(ordered_blocks):
            item = ordered_blocks[i]
            
            # Handle Text Blocks
            if "lines" in item:
                text = " ".join([s["text"] for l in item["lines"] for s in l["spans"]]).strip()
                if not text or "Reprint" in text:
                    i += 1
                    continue
                
                if re.match(config.RULES["HEADING_PATTERN"], text):
                    all_items.append({"type": "HEADING", "value": text})
                else:
                    all_items.append({"type": "CONTENT", "value": text})
            
            # Handle Diagram Markers
            elif item.get("type") == "DIAGRAM":
                all_items.append({"type": "CONTENT", "value": item["value"]})
                
                # Look-ahead for Caption (e.g., Fig 6.1)
                if i + 1 < len(ordered_blocks):
                    next_item = ordered_blocks[i+1]
                    if "lines" in next_item:
                        next_text = " ".join([s["text"] for l in next_item["lines"] for s in l["spans"]]).strip()
                        if re.match(config.RULES["FIGURE_PATTERN"], next_text, re.I):
                            all_items.append({"type": "CONTENT", "value": next_text})
                            i += 1
            i += 1

    doc.close()
    return all_items