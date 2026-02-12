import fitz

import re

import config



def classify_and_clean():

    doc = fitz.open(config.PDF_PATH)

    all_items = []

   

    for page in doc:

        # 1. COLUMN PARTITIONING

        page_width = page.rect.width

        split_x = page_width * config.COLUMN_GAP_THRESHOLD

       

        page_dict = page.get_text("dict")

        blocks = page_dict["blocks"]

       

        left_col, right_col = [], []

        for b in blocks:

            # Sort into buckets to prevent horizontal reading

            if b["bbox"][0] < split_x:

                left_col.append(b)

            else:

                right_col.append(b)



        # Sort each column top-to-bottom

        left_col.sort(key=lambda x: x["bbox"][1])

        right_col.sort(key=lambda x: x["bbox"][1])

       

        ordered_blocks = left_col + right_col



        i = 0

        while i < len(ordered_blocks):

            b = ordered_blocks[i]

           

            if "lines" in b:

                # Reconstruct text

                text = " ".join([s["text"] for l in b["lines"] for s in l["spans"]]).strip()

               

                if not text or "Reprint" in text:

                    i += 1

                    continue



                # Identify if it's a structural heading or just content

                if re.match(config.RULES["HEADING_PATTERN"], text):

                    all_items.append({"type": "HEADING", "value": text})

                elif any(k in text for k in config.RULES["EXERCISE_KEYWORDS"]):

                    all_items.append({"type": "EXERCISE", "value": text})

                else:

                    # Everything else (Paragraphs, Captions) is just "CONTENT"

                    all_items.append({"type": "CONTENT", "value": text})

           

            else:

                # Handle Image Zones

                image_label = f"Image at {b['bbox']}"

                all_items.append({"type": "CONTENT", "value": image_label})

               

                # Look-ahead for captions to keep them next to the image

                if i + 1 < len(ordered_blocks):

                    next_b = ordered_blocks[i+1]

                    if "lines" in next_b:

                        next_text = " ".join([s["text"] for l in next_b["lines"] for s in l["spans"]]).strip()

                        if re.match(config.RULES["FIGURE_PATTERN"], next_text, re.I):

                            all_items.append({"type": "CONTENT", "value": next_text})

                            i += 1

            i += 1



    doc.close()

    return all_items