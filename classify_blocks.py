

import fitz
import re
import config

def is_inside_table(block_rect, table_rects):
    """Checks if a block's rectangle is inside any known table rectangle."""
    for t_rect in table_rects:
        # Check if the block intersects significantly with a table area
        if block_rect.intersects(t_rect):
            return True
    return False

def classify_and_clean():
    doc = fitz.open(config.PDF_PATH)
    all_classified_blocks = []
    
    page_width = doc[0].rect.width
    mid_x = page_width * config.COLUMN_GAP_THRESHOLD

    for page in doc:
        # --- NEW: Identify Table Boundaries ---
        table_rects = []
        if config.IGNORE_TABLES:
            tabs = page.find_tables() # Finds table structures geometrically
            table_rects = [t.bbox for t in tabs]

        blocks = page.get_text("blocks")
        left_col, right_col = [], []

        for b in blocks:
            x0, y0, x1, y1, text, block_no, block_type = b
            block_rect = fitz.Rect(x0, y0, x1, y1)

            # 1. Skip if empty or noise
            if not text.strip() or "Reprint" in text:
                continue
            
            # 2. NEW: Skip if the block is inside a table boundary
            if is_inside_table(block_rect, table_rects):
                continue
            
            # Sort into columns as before
            if x0 < mid_x:
                left_col.append(b)
            else:
                right_col.append(b)

        # Sort and Classify (Rest of the logic remains same)
        ordered_blocks = sorted(left_col, key=lambda x: x[1]) + sorted(right_col, key=lambda x: x[1])

        for b in ordered_blocks:
            text = b[4].strip()
            category = "PARAGRAPH"
            
            if re.match(config.RULES["HEADING_PATTERN"], text):
                category = "HEADING"
            elif text.lower().startswith(("fig.", "figure")):
                # ... figure logic ...
                category = "FIGURE"
            elif re.search(config.RULES["EQUATION_PATTERN"], text):
                # ... equation logic ...
                category = "EQUATION"
            elif any(k in text for k in config.RULES["EXERCISE_KEYWORDS"]):
                category = "EXERCISE"

            all_classified_blocks.append({"type": category, "text": text})

    doc.close()
    return all_classified_blocks