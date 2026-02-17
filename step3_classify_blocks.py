import fitz
import re
import config
import os


def extract_text_from_block(block):
    return " ".join(
        s["text"] for l in block["lines"] for s in l["spans"]
    ).strip()


def get_diagram_bbox(page, caption_block, split_x):
    caption_rect = fitz.Rect(caption_block[:4])
    is_left_col = caption_rect.x0 < split_x

    col_min_x = 0 if is_left_col else split_x
    col_max_x = split_x if is_left_col else page.rect.width

    search_area = fitz.Rect(col_min_x, caption_rect.y0 - 300, col_max_x, caption_rect.y0)

    drawings = [
        d["rect"] for d in page.get_drawings()
        if d["rect"].intersects(search_area)
        and d["rect"].x0 >= col_min_x
        and d["rect"].x1 <= col_max_x
    ]

    if not drawings:
        return None

    diagram_box = drawings[0]
    for d_rect in drawings[1:]:
        diagram_box |= d_rect

    return diagram_box + (-5, -5, 5, 5)


def classify_and_clean(pdf_path=None, image_output_dir=None):

    target_pdf = pdf_path if pdf_path else config.PDF_PATH
    # target_image_dir = image_output_dir if image_output_dir else config.IMAGE_DIR
    target_image_dir = os.path.join("processed_data", "knowledge_base", "extracted_images")


    doc = fitz.open(target_pdf)
    all_items = []

    if not os.path.exists(target_image_dir):
        os.makedirs(target_image_dir)

    mode = "theory"

    for page_num, page in enumerate(doc):

        split_x = page.rect.width * config.COLUMN_GAP_THRESHOLD
        page_height = page.rect.height

        blocks = page.get_text("blocks")
        page_dict = page.get_text("dict")
        raw_blocks = [b for b in page_dict["blocks"] if "lines" in b]

        # -------------------------------------------------
        # REMOVE HEADER & FOOTER (position-based)
        # -------------------------------------------------
        cleaned_blocks = []
        for b in raw_blocks:
            y0, y1 = b["bbox"][1], b["bbox"][3]

            if y0 < page_height * 0.08:  # top 8%
                continue
            if y1 > page_height * 0.92:  # bottom 8%
                continue

            cleaned_blocks.append(b)

        raw_blocks = cleaned_blocks
        raw_blocks.sort(key=lambda b: b["bbox"][1])

        # -------------------------------------------------
        # IMAGE EXTRACTION (unchanged)
        # -------------------------------------------------
        diagrams_on_page = []

        for b in blocks:
            text = b[4].strip().replace("\n", " ")
            match = re.search(config.RULES["FIGURE_PATTERN"], text, re.I)

            if match:
                area = get_diagram_bbox(page, b, split_x)
                if area:
                    fig_id = match.group(1).replace('.', '_')
                    img_filename = f"fig_{fig_id}.png"
                    img_path = os.path.join(target_image_dir, img_filename)

                    pix = page.get_pixmap(clip=area, matrix=fitz.Matrix(3, 3))
                    pix.save(img_path)

                    diagrams_on_page.append({
                        "bbox": area,
                        "path": img_path,
                        "caption": text,
                        "is_left": area.x0 < split_x
                    })

        # -------------------------------------------------
        # IF ALREADY IN END MODE → SIMPLE LINEAR PARSE
        # -------------------------------------------------
        if mode != "theory":

            for b in raw_blocks:
                text = extract_text_from_block(b)
                if text:
                    all_items.append({"type": "CONTENT", "value": text})

            continue

        # -------------------------------------------------
        # DETECT SUMMARY POSITION (VERTICAL SPLIT)
        # -------------------------------------------------
        split_index = None
        trigger_keyword = None

        for i, b in enumerate(raw_blocks):
            text = extract_text_from_block(b).upper().strip()
            if text in config.RULES["END_SECTION_KEYWORDS"]:
                split_index = i
                trigger_keyword = text
                break

        if split_index is not None:
            theory_blocks = raw_blocks[:split_index]
            end_blocks = raw_blocks[split_index:]
            mode = "end_section"
        else:
            theory_blocks = raw_blocks
            end_blocks = []

        # -------------------------------------------------
        # PROCESS THEORY BLOCKS (TWO COLUMN)
        # -------------------------------------------------
        left_col, right_col = [], []
        processed_captions = [d["caption"] for d in diagrams_on_page]

        for b in theory_blocks:

            text = extract_text_from_block(b)

            if not text or text in processed_captions:
                continue

            if b["bbox"][0] < split_x:
                left_col.append(b)
            else:
                right_col.append(b)

        for d in diagrams_on_page:
            normalized_path = d["path"].replace("\\", "/")
            marker = {
                "type": "DIAGRAM",
                "bbox": d["bbox"],
                "value": f"[IMAGE: {normalized_path}]",
                "caption": d["caption"]
            }
            if d["is_left"]:
                left_col.append(marker)
            else:
                right_col.append(marker)

        merged = left_col + right_col
        merged.sort(key=lambda x: x["bbox"][1])

        for item in merged:
            if isinstance(item, dict) and "lines" in item:
                text = extract_text_from_block(item)
                if text:
                    itype = "HEADING" if re.match(config.RULES["HEADING_PATTERN"], text) else "CONTENT"
                    all_items.append({"type": itype, "value": text})

            elif isinstance(item, dict) and item.get("type") == "DIAGRAM":
                all_items.append({"type": "CONTENT", "value": item["value"]})
                all_items.append({"type": "CONTENT", "value": item["caption"]})

        # -------------------------------------------------
        # PROCESS END BLOCKS (SINGLE COLUMN)
        # -------------------------------------------------
        for b in end_blocks:
            text = extract_text_from_block(b)
            if text:
                all_items.append({"type": "CONTENT", "value": text})

    page_count = doc.page_count
    print(f"[DEBUG] Processed {page_count} pages.")
    print(f"[DEBUG] Total extracted items: {len(all_items)}")

    doc.close()
    return all_items
