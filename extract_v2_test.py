import fitz
import re
import os
import json
import config

def get_12th_diagram_bbox(page, label_rect):
    """
    Grade 12 Specific: Prioritizes Bitmap images and expands to catch labels.
    """
    # 1. Broad Search Area: Projected 450 points UP from the 'Fig.' label
    # Expanded width to 500pts to handle full-page 12th grade diagrams
    search_area = fitz.Rect(0, label_rect.y0 - 450, page.rect.width, label_rect.y0)
    
    elements = []
    
    # 2. CAPTURE BITMAPS: Grade 12 uses real photos/renders
    for img in page.get_images(full=True):
        for img_rect in page.get_image_rects(img[0]):
            if img_rect.intersects(search_area):
                elements.append(img_rect)
    
    # 3. CAPTURE VECTORS: Lines, arrows, and shaded boxes
    drawings = [d["rect"] for d in page.get_drawings() if d["rect"].intersects(search_area)]
    elements.extend(drawings)

    if not elements:
        return None
        
    # 4. PROXIMITY SORT: Start merging from the label UPWARD
    elements.sort(key=lambda r: r.y1, reverse=True)
    
    # 5. GREEDY UNION: Using a 60pt gap to bridge Grade 12 labels/diagrams
    diagram_box = elements[0]
    for i in range(1, len(elements)):
        # Skip horizontal page dividers
        if elements[i].width > (page.rect.width * 0.8):
            continue
            
        vertical_gap = abs(elements[i].y1 - diagram_box.y0)
        if vertical_gap < 60: 
            diagram_box |= elements[i]
        else:
            break
            
    # 6. QUALITY CHECK
    if diagram_box.height < 30 or diagram_box.width < 30:
        return None

    return diagram_box + (-5, -5, 5, 5)

def test_extraction_12th(pdf_name, output_image_dir):
    doc = fitz.open(pdf_name)
    os.makedirs(output_image_dir, exist_ok=True)
    all_items = []

    with open("extraction_debug.log", "w", encoding="utf-8") as debug:
        for page_num, page in enumerate(doc):
            debug.write(f"\n--- Processing Page {page_num + 1} ---\n")
            
            # PASS 1: Surgical Search for Figure Labels
            # Grade 12 labels are often 'floating' text; search_for is most reliable
            fig_instances = page.search_for("Fig.")
            processed_ids = set()
            diagram_bboxes = []

            for inst in fig_instances:
                # Look slightly right to catch the number (e.g. 2.1)
                num_area = inst + (0, 0, 50, 0)
                num_text = page.get_text("text", clip=num_area).strip()
                match = re.search(r'(\d+\.\d+)', num_text)
                
                if match:
                    fig_id_str = match.group(1)
                    if fig_id_str in processed_ids: continue
                    
                    debug.write(f"Found Label: Fig. {fig_id_str}\n")
                    
                    area = get_12th_diagram_bbox(page, inst)
                    if area:
                        fig_id = fig_id_str.replace('.', '_')
                        img_path = os.path.join(output_image_dir, f"fig_{fig_id}.png")
                        
                        # High-Res Render (3.0 zoom)
                        page.get_pixmap(clip=area, matrix=fitz.Matrix(3,3)).save(img_path)
                        
                        diagram_bboxes.append(area)
                        processed_ids.add(fig_id_str)
                        all_items.append({
                            "type": "CONTENT", 
                            "value": f"[IMAGE: {img_path.replace(os.sep, '/')}]",
                            "page": page_num + 1
                        })
                        debug.write(f"   Saved: {img_path}\n")

            # PASS 2: Text Blocks
            blocks = page.get_text("blocks")
            blocks.sort(key=lambda b: b[1])
            for b in blocks:
                b_rect = fitz.Rect(b[:4])
                text = b[4].strip().replace("\n", " ")
                if not text or "Reprint" in text: continue
                
                # Skip text inside diagrams
                if any(b_rect.intersects(db) for db in diagram_bboxes):
                    if "Fig." in text: # But keep the caption itself
                        all_items.append({"type": "CONTENT", "value": text, "page": page_num + 1})
                    continue
                
                itype = "HEADING" if re.match(r'^\d+\.\d+', text) else "CONTENT"
                all_items.append({"type": itype, "value": text, "page": page_num + 1})

    doc.close()
    return all_items

def build_json_v2(extracted_items):

    structure = {
        "chapters": []
    }

    current_chapter = None
    current_chapter_number = None
    curr_section = None
    curr_subsection = None
    mode = "theory"

    for item in extracted_items:
        val = item["value"].strip()
        upper_val = val.upper()

        # --------------------------------------------------
        # 1️⃣ CHAPTER DETECTION VIA X.1 INTRODUCTION
        # --------------------------------------------------
        chapter_match = re.match(r'^(\d+)\.1\s+', val)

        if item["type"] == "HEADING" and chapter_match:

            chapter_number = chapter_match.group(1)

            # If new chapter number detected
            if chapter_number != current_chapter_number:

                current_chapter_number = chapter_number

                current_chapter = {
                    "grade": "12",
                    "chapter_number": chapter_number,
                    "chapter_title": f"Chapter {chapter_number}",
                    "sections": [],
                    "summary": [],
                    "points_to_ponder": [],
                    "exercises": []
                }

                structure["chapters"].append(current_chapter)

                curr_section = None
                curr_subsection = None
                mode = "theory"

        # Skip anything before first chapter
        if current_chapter is None:
            continue

        # --------------------------------------------------
        # 2️⃣ MODE SWITCHING
        # --------------------------------------------------
        if upper_val == "SUMMARY":
            mode = "summary"
            continue

        if upper_val == "POINTS TO PONDER":
            mode = "points_to_ponder"
            continue

        if upper_val == "EXERCISES":
            mode = "exercises"
            continue

        # --------------------------------------------------
        # 3️⃣ SPECIAL SECTIONS
        # --------------------------------------------------
        if mode == "summary":
            current_chapter["summary"].append(val)
            continue

        if mode == "points_to_ponder":
            current_chapter["points_to_ponder"].append(val)
            continue

        if mode == "exercises":
            current_chapter["exercises"].append(val)
            continue

        # --------------------------------------------------
        # 4️⃣ THEORY MODE
        # --------------------------------------------------
        if item["type"] == "HEADING":

            # Section (e.g., 2.3 Electric Flux)
            section_match = re.match(r'^(\d+\.\d+)\s+(.*)', val)

            if section_match:
                curr_section = {
                    "section_number": section_match.group(1),
                    "section_title": section_match.group(2),
                    "content": [],
                    "subsections": []
                }
                current_chapter["sections"].append(curr_section)
                curr_subsection = None
                continue

            # Subsection (e.g., 2.3.1 Something)
            subsection_match = re.match(r'^(\d+\.\d+\.\d+)\s+(.*)', val)

            if subsection_match:
                curr_subsection = {
                    "subsection_number": subsection_match.group(1),
                    "subsection_title": subsection_match.group(2),
                    "content": []
                }
                if curr_section:
                    curr_section["subsections"].append(curr_subsection)
                continue

        # --------------------------------------------------
        # 5️⃣ CONTENT PLACEMENT
        # --------------------------------------------------
        target = curr_subsection if curr_subsection else curr_section
        if target:
            target["content"].append(val)

    return structure


# --- CRITICAL ADDITION: THE BRIDGE FUNCTION ---
def process_pdf_v2(pdf_path, image_output_dir):
    """
    This is the function pipeline.py is looking for.
    It runs the extraction and then builds the JSON hierarchy.
    """
    raw_items = test_extraction_12th(pdf_path, output_image_dir=image_output_dir)
    structure = build_json_v2(raw_items)
    return structure

if __name__ == "__main__":
    pdf_input = "Physics-12 1-45-106.pdf" 
    if os.path.exists(pdf_input):
        print("Test run initiated...")
        result = process_pdf_v2(pdf_input, "extract_images")
        with open("physics2_structure.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4)
        print("Check 'physics2_structure.json' for results.")