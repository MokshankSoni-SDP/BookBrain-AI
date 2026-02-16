import fitz
import re
import os
import json

def get_12th_diagram_bbox(page, caption_rect):
    """
    Greedy Lasso for Grade 12: Bridges larger gaps between diagram parts.
    """
    # 1. Broad Search Area: proyecting 450 points UP (Grade 12 diagrams can be tall)
    # Using a 250pt horizontal buffer to capture labels on the sides
    search_x0 = max(0, caption_rect.x0 - 250)
    search_x1 = min(page.rect.width, caption_rect.x1 + 250)
    search_area = fitz.Rect(search_x0, caption_rect.y0 - 450, search_x1, caption_rect.y0)
    
    # 2. Gather ALL visual elements (Drawings + Images)
    elements = [d["rect"] for d in page.get_drawings() if d["rect"].intersects(search_area)]
    for img in page.get_images(full=True):
        for img_rect in page.get_image_rects(img[0]):
            if img_rect.intersects(search_area):
                elements.append(img_rect)

    if not elements:
        return None
        
    # 3. Sort elements Bottom-to-Top (Nearest to caption first)
    elements.sort(key=lambda r: r.y1, reverse=True)
    
    # 4. Greedy Union: Higher gap threshold (50pts) to bridge Grade 12 labels
    diagram_box = elements[0]
    for i in range(1, len(elements)):
        # Ignore extremely wide lines (likely section dividers)
        if elements[i].width > (page.rect.width * 0.85):
            continue
            
        # Check gap between the top of our current box and the bottom of the next element
        vertical_gap = abs(elements[i].y1 - diagram_box.y0)
        
        if vertical_gap < 50: 
            diagram_box |= elements[i]
        else:
            # We hit a significant gap (likely the start of text above)
            break
            
    # 5. Safety: Discard tiny noise or page-sized errors
    if diagram_box.height < 30 or diagram_box.width < 30:
        return None
    if diagram_box.height > 600:
        return None

    return diagram_box + (-10, -10, 10, 10) # Comfortable Padding

def test_extraction_12th(pdf_name, output_image_dir="extract_images"):
    doc = fitz.open(pdf_name)
    os.makedirs(output_image_dir, exist_ok=True)
    all_items = []

    print(f"Starting extraction for: {pdf_name}")

    for page_num, page in enumerate(doc):
        # 1. PRECISE FIGURE SEARCH: Find exact coordinates of all figure labels
        # This prevents missing labels buried inside larger text blocks
        fig_labels = page.search_for("Fig.")
        processed_ids = set()
        diagram_bboxes_on_page = []

        for label_rect in fig_labels:
            # Look slightly to the right of 'Fig.' to find the ID (e.g., 2.1)
            context_area = label_rect + (0, 0, 60, 0) 
            context_text = page.get_text("text", clip=context_area).strip()
            
            match = re.search(r'(\d+\.\d+)', context_text)
            if match:
                fig_id_str = match.group(1)
                if fig_id_str in processed_ids: 
                    continue
                
                # Use the new precise BBox logic centered on the label
                area = get_12th_diagram_bbox(page, label_rect)
                
                if area:
                    fig_id = fig_id_str.replace('.', '_')
                    img_filename = f"fig_{fig_id}.png"
                    img_path = os.path.join(output_image_dir, img_filename)
                    
                    # 300 DPI High-Quality Render
                    page.get_pixmap(clip=area, matrix=fitz.Matrix(3,3)).save(img_path)
                    
                    # Store diagram area to avoid treating it as text later
                    diagram_bboxes_on_page.append(area)
                    processed_ids.add(fig_id_str)
                    
                    # Add image marker to items
                    all_items.append({
                        "type": "CONTENT", 
                        "value": f"[IMAGE: {img_path.replace(os.sep, '/')}]",
                        "page": page_num + 1
                    })

        # 2. TEXT EXTRACTION: Standard top-to-bottom flow
        blocks = page.get_text("blocks")
        blocks.sort(key=lambda b: b[1]) 

        for b in blocks:
            block_rect = fitz.Rect(b[:4])
            text = b[4].strip().replace("\n", " ")
            
            # Skip if text is empty, a reprint notice, or overlaps with a saved diagram
            if not text or "Reprint" in text: 
                continue
            if any(block_rect.intersects(db) for db in diagram_bboxes_on_page):
                # If this block is the actual caption we found earlier, include it
                if "Fig." in text:
                    all_items.append({"type": "CONTENT", "value": text, "page": page_num + 1})
                continue

            # Identify Headings vs regular content
            if re.match(r'^\d+\.\d+', text):
                all_items.append({"type": "HEADING", "value": text, "page": page_num + 1})
            else:
                all_items.append({"type": "CONTENT", "value": text, "page": page_num + 1})

    doc.close()
    return all_items

def build_json_v2(extracted_items, output_json="physics2_structure.json"):
    """Hierarchy builder logic to create the JSON structure."""
    structure = {
        "chapter_title": "Physics Grade 12 - Chapter 1",
        "sections": []
    }
    
    curr_section = None
    curr_subsection = None

    for item in extracted_items:
        val = item["value"]
        
        # Section Detection (e.g., 1.1 Electric Charge)
        if item["type"] == "HEADING" and re.match(r'^\d+\.\d+\s', val):
            curr_section = {
                "section_number": val.split()[0],
                "section_title": " ".join(val.split()[1:]),
                "content": [],
                "subsections": []
            }
            structure["sections"].append(curr_section)
            curr_subsection = None
            
        # Subsection Detection (e.g., 1.1.1)
        elif item["type"] == "HEADING" and re.match(r'^\d+\.\d+\.\d+', val):
            curr_subsection = {
                "subsection_number": val.split()[0],
                "subsection_title": " ".join(val.split()[1:]),
                "content": []
            }
            if curr_section:
                curr_section["subsections"].append(curr_subsection)
        else:
            target = curr_subsection if curr_subsection else curr_section
            if target:
                target["content"].append(val)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(structure, f, indent=4)
    print(f"Successfully saved to: {output_json}")

if __name__ == "__main__":
    pdf_input = "Physics-12 1-45-106.pdf" 
    if os.path.exists(pdf_input):
        print(f"File found! Starting extraction for {pdf_input}...")
        raw_items = test_extraction_12th(pdf_input)
        build_json_v2(raw_items)
    else:
        print(f"Error: {pdf_input} not found.")