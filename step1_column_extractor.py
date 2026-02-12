import fitz
import config
import re
import statistics
import os


class ProductionColumnExtractor:

    def __init__(self, pdf_path):
        self.doc = fitz.open(pdf_path)

    # ------------------------------------------------
    # Detect math-heavy blocks
    # ------------------------------------------------
    def is_math_block(self, text):
        text = text.strip()
        if not text:
            return False

        # Count math-like symbols
        math_chars = r'[=+\-×*/^∑∆τ∫√()]'
        math_count = len(re.findall(math_chars, text))

        # Ratio check
        if len(text) > 0 and math_count / len(text) > 0.25:
            return True

        # Variable-like pattern
        if re.search(r'\b[a-zA-Z]\s*=\s*', text):
            return True

        # Summation / integral pattern
        if re.search(r'(∑|∫)', text):
            return True

        return False

    def merge_vertical_math(self, blocks, y_threshold=12):

        merged = []
        i = 0

        while i < len(blocks):
            current = blocks[i]

            if current["type"] == "text" and self.is_math_block(current["content"]):
                combined = current["content"]
                j = i + 1

                while (
                    j < len(blocks)
                    and blocks[j]["type"] == "text"
                    and self.is_math_block(blocks[j]["content"])
                    and abs(blocks[j]["y0"] - blocks[j - 1]["y0"]) < y_threshold
                ):
                    combined += " " + blocks[j]["content"]
                    j += 1

                merged.append({
                    "type": "text",
                    "content": combined
                })

                i = j
            else:
                merged.append(current)
                i += 1

        return merged


    # ------------------------------------------------
    # Noise filtering
    # ------------------------------------------------
    def is_noise(self, text, width):
        text = text.strip()

        if not text:
            return True

        # Single letters like (a), (b)
        if re.fullmatch(r'\(?[a-zA-Z]\)?', text):
            return True

        # Pure symbols only
        if not re.search(r'[a-zA-Z]', text) and len(text) < 10:
            return True

        # Very tiny junk blocks
        if len(text) < 3:
            return True

        return False


    # ------------------------------------------------
    # Extract structured blocks using span-level control
    # ------------------------------------------------
    def extract_page_blocks(self, page):

        page_dict = page.get_text("dict")
        blocks = []

        for block in page_dict["blocks"]:

            # ---------------- TEXT BLOCK ----------------
            if block["type"] == 0:
                lines = []

                for line in block["lines"]:
                    line_text = ""
                    prev_x = None

                    for span in line["spans"]:
                        span_text = span["text"]

                        # Insert space if spans are separated
                        if prev_x is not None:
                            if span["bbox"][0] - prev_x > 2:
                                line_text += " "

                        line_text += span_text
                        prev_x = span["bbox"][2]

                    lines.append(line_text.strip())

                full_text = "\n".join(lines).strip()

                if not full_text:
                    continue

                if not re.search(r'[a-zA-Z]', full_text):
                    continue

                x0, y0, x1, y1 = block["bbox"]
                width = x1 - x0

                # Header/Footer filtering
                if y0 < config.HEADER_CUTOFF or y0 > config.FOOTER_CUTOFF:
                    continue

                if self.is_noise(full_text, width):
                    continue

                blocks.append({
                    "type": "text",
                    "content": full_text,
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "centroid_x": (x0 + x1) / 2
                })

        return blocks

    # ------------------------------------------------
    # Column ordering
    # ------------------------------------------------
    def order_by_columns(self, blocks, page):

        if not blocks:
            return []

        page_width = page.rect.width
        split_x = page_width / 2

        left = []
        right = []

        for b in blocks:
            if b["centroid_x"] < split_x:
                left.append(b)
            else:
                right.append(b)

        left.sort(key=lambda x: x["y0"])
        right.sort(key=lambda x: x["y0"])

        return left + right


    # ------------------------------------------------
    # Merge math blocks
    # ------------------------------------------------
    def merge_math_blocks(self, blocks):

        merged = []
        i = 0

        while i < len(blocks):

            current = blocks[i]

            if current["type"] == "text" and self.is_math_block(current["content"]):
                combined = current["content"]
                j = i + 1

                while (
                    j < len(blocks)
                    and blocks[j]["type"] == "text"
                    and self.is_math_block(blocks[j]["content"])
                ):
                    combined += " " + blocks[j]["content"]
                    j += 1

                merged.append({
                    "type": "text",
                    "content": combined
                })

                i = j
            else:
                merged.append({
                    "type": current["type"],
                    "content": current["content"]
                })
                i += 1

        return merged

    def is_centered(self, block, page_width, tolerance=0.15):
        center = page_width / 2
        block_center = block["centroid_x"]
        return abs(block_center - center) < (page_width * tolerance)

    def is_short_math_line(self, text):
        text = text.strip()
        if len(text) < 20 and self.is_math_block(text):
            return True
        return False

    def merge_equation_stacks(self, blocks, page_width, y_threshold=15):

        merged = []
        i = 0

        while i < len(blocks):
            current = blocks[i]

            if (
                current["type"] == "text"
                and self.is_short_math_line(current["content"])
                and self.is_centered(current, page_width)
            ):

                combined = current["content"]
                j = i + 1

                while (
                    j < len(blocks)
                    and blocks[j]["type"] == "text"
                    and self.is_short_math_line(blocks[j]["content"])
                    and self.is_centered(blocks[j], page_width)
                    and abs(blocks[j]["y0"] - blocks[j-1]["y0"]) < y_threshold
                ):
                    combined += " " + blocks[j]["content"]
                    j += 1

                merged.append({
                    "type": "text",
                    "content": combined
                })

                i = j

            else:
                merged.append(current)
                i += 1

        return merged

    # ------------------------------------------------
    # Merge split headings like:
    # 6.1
    # Introduction
    # ------------------------------------------------
    def merge_headings(self, blocks):

        merged = []
        i = 0

        while i < len(blocks):

            current = blocks[i]

            if (
                current["type"] == "text"
                and re.match(r'^\d+\.\d+$', current["content"].strip())
                and i + 1 < len(blocks)
                and blocks[i + 1]["type"] == "text"
            ):
                combined = current["content"].strip() + " " + blocks[i + 1]["content"].strip()
                merged.append({
                    "type": "text",
                    "content": combined
                })
                i += 2
                continue

            merged.append(current)
            i += 1

        return merged

    # ------------------------------------------------
    # Main extraction
    # ------------------------------------------------
    def extract_clean_text(self):

        all_blocks = []

        for page_number, page in enumerate(self.doc):

            page_blocks = self.extract_page_blocks(page)
            ordered = self.order_by_columns(page_blocks,page)
            merged_math = self.merge_math_blocks(ordered)
            merged_vertical = self.merge_vertical_math(merged_math)
            merged_equations = self.merge_equation_stacks(
                merged_vertical,
                page.rect.width
            )

            final_blocks = self.merge_headings(merged_vertical)

            print(f"\n--- Page {page_number + 1} ---")
            for b in final_blocks[:5]:
                if b["type"] == "text":
                    print(b["content"][:120])
                else:
                    print("[IMAGE BLOCK]")
                print("-" * 40)

            all_blocks.extend(final_blocks)

        return all_blocks





# ------------------------------------------------
# Run
# ------------------------------------------------
if __name__ == "__main__":
    extractor = ProductionColumnExtractor(config.PDF_PATH)
    blocks = extractor.extract_clean_text()
    print(f"\nTotal extracted blocks: {len(blocks)}")
