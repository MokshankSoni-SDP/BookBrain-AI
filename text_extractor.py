import fitz
import config
import re

class FinalCleanExtractor:
    def __init__(self, pdf_path):
        self.doc = fitz.open(pdf_path)

    # ------------------------------
    # Filtering Rules
    # ------------------------------
    def is_noise(self, text, width):
        text = text.strip()

        if not text:
            return True

        # Remove very short fragments
        if len(text) < 4:
            return True

        # Remove single letters or (a), (b)
        if re.match(r'^\(?[a-zA-Z]\)?$', text):
            return True

        # Remove equation-only fragments (mostly symbols)
        symbol_ratio = sum(1 for c in text if not c.isalnum()) / len(text)
        if symbol_ratio > 0.6:
            return True

        # Remove narrow sidebar blocks
        if width < 100:
            return True

        # Remove Table of Contents style blocks
        if re.findall(r'6\.\d+', text) and len(text.split()) < 6:
            return True

        return False

    # ------------------------------
    # Merge split headings like:
    # 6.1
    # Introduction
    # ------------------------------
    def merge_split_headings(self, blocks):
        merged = []
        i = 0

        while i < len(blocks):
            current = blocks[i]["text"].strip()

            # If block is just section number like "6.1"
            if re.match(r'^6\.\d+$', current):
                if i + 1 < len(blocks):
                    next_text = blocks[i + 1]["text"].strip()
                    # Merge if next line looks like title
                    if next_text and not re.match(r'^6\.', next_text):
                        merged_text = current + " " + next_text
                        merged.append({"text": merged_text})
                        i += 2
                        continue

            merged.append({"text": current})
            i += 1

        return merged

    # ------------------------------
    # Extract Clean Page Blocks
    # ------------------------------
    def extract_page_blocks(self, page):
        raw_blocks = page.get_text("blocks")
        page_width = page.rect.width

        clean_blocks = []

        for block in raw_blocks:
            x0, y0, x1, y1, text, block_no = block[:6]

            width = x1 - x0
            text = text.strip()

            # Remove header/footer
            if y0 < config.HEADER_CUTOFF or y0 > config.FOOTER_CUTOFF:
                continue

            if self.is_noise(text, width):
                continue

            clean_blocks.append({
                "text": text,
                "x0": x0,
                "y0": y0
            })

        # Sort by vertical first, then horizontal
        clean_blocks.sort(key=lambda b: (b["y0"], b["x0"]))

        # Merge split headings
        clean_blocks = self.merge_split_headings(clean_blocks)

        return clean_blocks

    # ------------------------------
    # Extract Full Chapter Text
    # ------------------------------
    def extract_clean_text(self):
        all_blocks = []

        for page_number, page in enumerate(self.doc):
            page_blocks = self.extract_page_blocks(page)

            print(f"\n--- Page {page_number + 1} ---")

            for block in page_blocks[:6]:
                print(block["text"][:120])
                print("-" * 40)

            all_blocks.extend(page_blocks)

        return all_blocks


if __name__ == "__main__":
    extractor = FinalCleanExtractor(config.PDF_PATH)
    blocks = extractor.extract_clean_text()

    print(f"\nTotal cleaned blocks: {len(blocks)}")
