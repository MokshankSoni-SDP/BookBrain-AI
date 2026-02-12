import fitz
import config
import os

class ColumnAwareExtractor:
    def __init__(self, pdf_path):
        self.doc = fitz.open(pdf_path)

    def extract_page_blocks(self, page):
        """
        Extract blocks from page and reconstruct correct reading order.
        """
        blocks = page.get_text("blocks")  # paragraph-level blocks
        page_width = page.rect.width
        split_x = page_width / 2

        header_cutoff = config.HEADER_CUTOFF
        footer_cutoff = config.FOOTER_CUTOFF

        left_column = []
        right_column = []

        for block in blocks:
            x0, y0, x1, y1, text, block_no = block[:6]

            text = text.strip()
            if not text:
                continue

            # Skip header/footer noise
            if y0 < header_cutoff or y0 > footer_cutoff:
                continue

            # Divide into columns dynamically
            if x0 < split_x:
                left_column.append((y0, text))
            else:
                right_column.append((y0, text))

        # Sort each column top-to-bottom
        left_column.sort(key=lambda x: x[0])
        right_column.sort(key=lambda x: x[0])

        # Merge reading order (Left first, then Right)
        ordered_blocks = [text for _, text in left_column]
        ordered_blocks += [text for _, text in right_column]

        return ordered_blocks

    def extract_chapter_text(self):
        """
        Extract all pages and reconstruct reading order.
        """
        all_text_blocks = []

        for page_number, page in enumerate(self.doc):
            page_blocks = self.extract_page_blocks(page)

            print(f"\n--- Page {page_number + 1} ---")
            for block in page_blocks[:5]:  # print first few for verification
                print(block[:100])
                print("-" * 40)

            all_text_blocks.extend(page_blocks)

        return all_text_blocks


if __name__ == "__main__":
    extractor = ColumnAwareExtractor(config.PDF_PATH)
    blocks = extractor.extract_chapter_text()

    print(f"\nTotal reconstructed blocks: {len(blocks)}")
