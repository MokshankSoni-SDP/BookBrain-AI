import pdfplumber
import fitz  # PyMuPDF
import pymupdf4llm
import re
from pathlib import Path
from PIL import Image
import io


class ImprovedColumnExtractor:
    """
    Enhanced PDF extractor with proper column handling and formula extraction
    """

    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.plumber_pdf = pdfplumber.open(pdf_path)
        self.fitz_doc = fitz.open(pdf_path)
        self.image_output_dir = Path("extracted_images")
        self.image_output_dir.mkdir(exist_ok=True)
        
        # Store extracted content in order
        self.page_contents = []

    def detect_columns(self, page):
        """Detect if page has columns and return column boundaries"""
        width = page.width
        height = page.height
        
        # For standard two-column layout
        # Check text distribution to confirm columns
        left_bbox = (0, 0, width/2, height)
        right_bbox = (width/2, 0, width, height)
        
        left_text = page.within_bbox(left_bbox).extract_text()
        right_text = page.within_bbox(right_bbox).extract_text()
        
        # If both halves have substantial text, it's two columns
        if left_text and right_text and len(left_text) > 50 and len(right_text) > 50:
            return [left_bbox, right_bbox]
        
        # Single column or special layout
        return [(0, 0, width, height)]

    def extract_formulas_from_text(self, text):
        """Extract and mark formulas in text using pattern matching"""
        # Patterns for common formula indicators
        formula_patterns = [
            r'\b[A-Z][a-z]?\s*=\s*[^,\.]+',  # Basic equations like F = ma
            r'\d+\.\d+[a-z]?\)',  # Equation numbers like (6.1a)
            r'[∑∫∂∇√±×÷≈≠≤≥∆]',  # Math symbols
            r'[α-ωΑ-Ω]',  # Greek letters
        ]
        
        has_formula = any(re.search(pattern, text) for pattern in formula_patterns)
        return has_formula

    def extract_text_blocks(self, page, bbox):
        """Extract text blocks with position information"""
        # Use pdfplumber's layout mode for better structure
        page_crop = page.within_bbox(bbox)
        
        # Extract words with positions
        words = page_crop.extract_words(
            x_tolerance=3,
            y_tolerance=3,
            keep_blank_chars=False,
            use_text_flow=True
        )
        
        if not words:
            return []
        
        # Group words into lines based on y-coordinate
        lines = []
        current_line = []
        current_y = None
        y_tolerance = 5
        
        for word in words:
            word_y = word['top']
            
            if current_y is None or abs(word_y - current_y) < y_tolerance:
                current_line.append(word)
                current_y = word_y if current_y is None else current_y
            else:
                if current_line:
                    lines.append(current_line)
                current_line = [word]
                current_y = word_y
        
        if current_line:
            lines.append(current_line)
        
        # Convert lines to text blocks
        blocks = []
        current_block = []
        
        for line in lines:
            line_text = ' '.join(word['text'] for word in line)
            
            # Check if this is a heading (larger font, bold, etc.)
            is_heading = self.is_heading(line)
            
            # Check for paragraph break (larger vertical gap)
            if current_block and len(blocks) > 0:
                prev_line = lines[lines.index(line) - 1]
                gap = line[0]['top'] - prev_line[-1]['bottom']
                
                if gap > 10 or is_heading:  # New paragraph
                    blocks.append({
                        'type': 'text',
                        'content': ' '.join(current_block),
                        'is_formula': self.extract_formulas_from_text(' '.join(current_block)),
                        'y_position': lines[len(blocks)][0]['top'] if blocks else line[0]['top']
                    })
                    current_block = []
            
            current_block.append(line_text)
        
        if current_block:
            blocks.append({
                'type': 'text',
                'content': ' '.join(current_block),
                'is_formula': self.extract_formulas_from_text(' '.join(current_block)),
                'y_position': lines[-1][0]['top']
            })
        
        return blocks

    def is_heading(self, line):
        """Detect if line is a heading based on font size and style"""
        if not line:
            return False
        
        # Check for section numbering pattern
        first_text = line[0]['text']
        if re.match(r'^\d+\.\d+', first_text):
            return True
        
        # Check if all caps (common for headings)
        full_text = ' '.join(word['text'] for word in line)
        if full_text.isupper() and len(full_text) > 5:
            return True
        
        return False

    def extract_images_with_context(self, page_num):
        """Extract images with their captions and context"""
        page = self.fitz_doc[page_num]
        plumber_page = self.plumber_pdf.pages[page_num]
        
        image_list = page.get_images()
        extracted_images = []
        
        for img_index, img in enumerate(image_list):
            try:
                xref = img[0]
                base_image = self.fitz_doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                # Save image
                image_filename = f"page_{page_num + 1}_img_{img_index + 1}.png"
                image_path = self.image_output_dir / image_filename
                
                with open(image_path, "wb") as img_file:
                    img_file.write(image_bytes)
                
                # Get image position
                img_rect = page.get_image_rects(xref)[0] if page.get_image_rects(xref) else None
                
                # Extract caption by looking for text near image
                caption = self.extract_caption_near_image(plumber_page, img_rect, page_num)
                
                extracted_images.append({
                    'type': 'image',
                    'filename': image_filename,
                    'path': str(image_path),
                    'caption': caption,
                    'page': page_num + 1,
                    'y_position': img_rect.y0 if img_rect else 0
                })
                
            except Exception as e:
                print(f"Error extracting image {img_index} from page {page_num + 1}: {e}")
                continue
        
        return extracted_images

    def extract_caption_near_image(self, page, img_rect, page_num):
        """Extract caption text near an image"""
        if not img_rect:
            return ""
        
        # Look for text below the image (typical caption location)
        caption_bbox = (
            img_rect.x0 - 20,  # Slightly wider than image
            img_rect.y1,  # Start from bottom of image
            img_rect.x1 + 20,
            img_rect.y1 + 100  # Look 100 points below
        )
        
        try:
            caption_area = page.within_bbox(caption_bbox)
            caption_text = caption_area.extract_text()
            
            if caption_text:
                # Clean up caption
                caption_text = caption_text.strip()
                # Look for Fig. X.XX pattern
                if re.search(r'Fig\.?\s*\d+\.\d+', caption_text, re.IGNORECASE):
                    return caption_text
        except:
            pass
        
        return ""

    def merge_content_by_position(self, text_blocks, images):
        """Merge text and images based on vertical position"""
        all_content = text_blocks + images
        
        # Sort by y_position
        all_content.sort(key=lambda x: x.get('y_position', 0))
        
        return all_content

    def extract_page_content(self, page_num):
        """Extract all content from a page in reading order"""
        page = self.plumber_pdf.pages[page_num]
        
        # Detect columns
        column_bboxes = self.detect_columns(page)
        
        all_blocks = []
        
        # Extract from each column in order (left to right)
        for bbox in column_bboxes:
            blocks = self.extract_text_blocks(page, bbox)
            all_blocks.extend(blocks)
        
        # Extract images
        images = self.extract_images_with_context(page_num)
        
        # Merge and sort by position
        merged_content = self.merge_content_by_position(all_blocks, images)
        
        return {
            'page_number': page_num + 1,
            'content': merged_content
        }

    def extract_all_pages(self):
        """Extract content from all pages"""
        all_pages_content = []
        
        for page_num in range(len(self.plumber_pdf.pages)):
            print(f"Processing page {page_num + 1}/{len(self.plumber_pdf.pages)}...")
            
            page_content = self.extract_page_content(page_num)
            all_pages_content.append(page_content)
        
        return all_pages_content

    def close(self):
        """Close PDF documents"""
        self.plumber_pdf.close()
        self.fitz_doc.close()


if __name__ == "__main__":
    import config
    
    print("=" * 60)
    print("ENHANCED PDF CONTENT EXTRACTION")
    print("=" * 60)
    
    extractor = ImprovedColumnExtractor(config.PDF_PATH)
    pages_content = extractor.extract_all_pages()
    extractor.close()
    
    print(f"\n✓ Extracted content from {len(pages_content)} pages")
    print(f"✓ Images saved to: {extractor.image_output_dir}")


# import fitz
# import config
# import re
# import statistics
# import os


# class ProductionColumnExtractor:

#     def __init__(self, pdf_path):
#         self.doc = fitz.open(pdf_path)

#     # ------------------------------------------------
#     # Detect math-heavy blocks
#     # ------------------------------------------------
#     def is_math_block(self, text):
#         text = text.strip()
#         if not text:
#             return False

#         # Count math-like symbols
#         math_chars = r'[=+\-×*/^∑∆τ∫√()]'
#         math_count = len(re.findall(math_chars, text))

#         # Ratio check
#         if len(text) > 0 and math_count / len(text) > 0.25:
#             return True

#         # Variable-like pattern
#         if re.search(r'\b[a-zA-Z]\s*=\s*', text):
#             return True

#         # Summation / integral pattern
#         if re.search(r'(∑|∫)', text):
#             return True

#         return False

#     def merge_vertical_math(self, blocks, y_threshold=12):

#         merged = []
#         i = 0

#         while i < len(blocks):
#             current = blocks[i]

#             if current["type"] == "text" and self.is_math_block(current["content"]):
#                 combined = current["content"]
#                 j = i + 1

#                 while (
#                     j < len(blocks)
#                     and blocks[j]["type"] == "text"
#                     and self.is_math_block(blocks[j]["content"])
#                     and abs(blocks[j]["y0"] - blocks[j - 1]["y0"]) < y_threshold
#                 ):
#                     combined += " " + blocks[j]["content"]
#                     j += 1

#                 merged.append({
#                     "type": "text",
#                     "content": combined,
#                     "x0": current["x0"],
#                     "y0": current["y0"],
#                     "x1": current["x1"],
#                     "centroid_x": current["centroid_x"]
#                 })

#                 i = j
#             else:
#                 merged.append(current)
#                 i += 1

#         return merged


#     # ------------------------------------------------
#     # Noise filtering
#     # ------------------------------------------------
#     def is_noise(self, text, width):
#         text = text.strip()

#         if not text:
#             return True

#         # Single letters like (a), (b)
#         if re.fullmatch(r'\(?[a-zA-Z]\)?', text):
#             return True

#         # Pure symbols only
#         if not re.search(r'[a-zA-Z]', text) and len(text) < 10:
#             return True

#         # Very tiny junk blocks
#         if len(text) < 3:
#             return True

#         return False


#     # ------------------------------------------------
#     # Extract structured blocks using span-level control
#     # ------------------------------------------------
#     def extract_page_blocks(self, page):

#         page_dict = page.get_text("dict")
#         blocks = []

#         for block in page_dict["blocks"]:

#             # ---------------- TEXT BLOCK ----------------
#             if block["type"] == 0:
#                 lines = []

#                 for line in block["lines"]:
#                     line_text = ""
#                     prev_x = None

#                     for span in line["spans"]:
#                         span_text = span["text"]

#                         # Insert space if spans are separated
#                         if prev_x is not None:
#                             if span["bbox"][0] - prev_x > 2:
#                                 line_text += " "

#                         line_text += span_text
#                         prev_x = span["bbox"][2]

#                     lines.append(line_text.strip())

#                 full_text = "\n".join(lines).strip()

#                 if not full_text:
#                     continue

#                 if not re.search(r'[a-zA-Z]', full_text):
#                     continue

#                 x0, y0, x1, y1 = block["bbox"]
#                 width = x1 - x0

#                 # Header/Footer filtering
#                 if y0 < config.HEADER_CUTOFF or y0 > config.FOOTER_CUTOFF:
#                     continue

#                 if self.is_noise(full_text, width):
#                     continue

#                 blocks.append({
#                     "type": "text",
#                     "content": full_text,
#                     "x0": x0,
#                     "y0": y0,
#                     "x1": x1,
#                     "centroid_x": (x0 + x1) / 2
#                 })

#         return blocks

#     # ------------------------------------------------
#     # Column ordering
#     # ------------------------------------------------
#     def order_by_columns(self, blocks, page):

#         if not blocks:
#             return []

#         page_width = page.rect.width
#         split_x = page_width / 2

#         left = []
#         right = []

#         for b in blocks:
#             if b["centroid_x"] < split_x:
#                 left.append(b)
#             else:
#                 right.append(b)

#         left.sort(key=lambda x: x["y0"])
#         right.sort(key=lambda x: x["y0"])

#         return left + right


#     # ------------------------------------------------
#     # Merge math blocks
#     # ------------------------------------------------
#     def merge_math_blocks(self, blocks):

#         merged = []
#         i = 0

#         while i < len(blocks):

#             current = blocks[i]

#             if current["type"] == "text" and self.is_math_block(current["content"]):
#                 combined = current["content"]
#                 j = i + 1

#                 while (
#                     j < len(blocks)
#                     and blocks[j]["type"] == "text"
#                     and self.is_math_block(blocks[j]["content"])
#                 ):
#                     combined += " " + blocks[j]["content"]
#                     j += 1

#                 merged.append({
#                     "type": "text",
#                     "content": combined,
#                     "x0": current["x0"],
#                     "y0": current["y0"],
#                     "x1": current["x1"],
#                     "centroid_x": current["centroid_x"]
#                 })

#                 i = j
#             else:
#                 merged.append({
#                     "type": current["type"],
#                     "content": current["content"]
#                 })
#                 i += 1

#         return merged

#     def is_centered(self, block, page_width, tolerance=0.15):
#         center = page_width / 2
#         block_center = block["centroid_x"]
#         return abs(block_center - center) < (page_width * tolerance)

#     def is_short_math_line(self, text):
#         text = text.strip()
#         if len(text) < 20 and self.is_math_block(text):
#             return True
#         return False

#     def merge_equation_stacks(self, blocks, page_width, y_threshold=15):

#         merged = []
#         i = 0

#         while i < len(blocks):
#             current = blocks[i]

#             if (
#                 current["type"] == "text"
#                 and self.is_short_math_line(current["content"])
#                 and self.is_centered(current, page_width)
#             ):

#                 combined = current["content"]
#                 j = i + 1

#                 while (
#                     j < len(blocks)
#                     and blocks[j]["type"] == "text"
#                     and self.is_short_math_line(blocks[j]["content"])
#                     and self.is_centered(blocks[j], page_width)
#                     and abs(blocks[j]["y0"] - blocks[j-1]["y0"]) < y_threshold
#                 ):
#                     combined += " " + blocks[j]["content"]
#                     j += 1

#                 merged.append({
#                     "type": "text",
#                     "content": combined,
#                     "x0": current["x0"],
#                     "y0": current["y0"],
#                     "x1": current["x1"],
#                     "centroid_x": current["centroid_x"]
#                 })

#                 i = j

#             else:
#                 merged.append(current)
#                 i += 1

#         return merged

#     # ------------------------------------------------
#     # Merge split headings like:
#     # 6.1
#     # Introduction
#     # ------------------------------------------------
#     def merge_headings(self, blocks):

#         merged = []
#         i = 0

#         while i < len(blocks):

#             current = blocks[i]

#             if (
#                 current["type"] == "text"
#                 and re.match(r'^\d+\.\d+$', current["content"].strip())
#                 and i + 1 < len(blocks)
#                 and blocks[i + 1]["type"] == "text"
#             ):
#                 combined = current["content"].strip() + " " + blocks[i + 1]["content"].strip()
#                 merged.append({
#                     "type": "text",
#                     "content": combined
#                 })
#                 i += 2
#                 continue

#             merged.append(current)
#             i += 1

#         return merged

#     # ------------------------------------------------
#     # Main extraction
#     # ------------------------------------------------
#     def extract_clean_text(self):

#         all_blocks = []

#         for page_number, page in enumerate(self.doc):

#             page_blocks = self.extract_page_blocks(page)
#             ordered = self.order_by_columns(page_blocks,page)
#             merged_math = self.merge_math_blocks(ordered)
#             merged_vertical = self.merge_vertical_math(merged_math)
#             merged_equations = self.merge_equation_stacks(
#                 merged_vertical,
#                 page.rect.width
#             )

#             final_blocks = self.merge_headings(merged_equations)

#             print(f"\n--- Page {page_number + 1} ---")
#             for b in final_blocks[:5]:
#                 if b["type"] == "text":
#                     print(b["content"][:120])
#                 else:
#                     print("[IMAGE BLOCK]")
#                 print("-" * 40)

#             all_blocks.extend(final_blocks)

#         return all_blocks





# # ------------------------------------------------
# # Run
# # ------------------------------------------------
# if __name__ == "__main__":
#     extractor = ProductionColumnExtractor(config.PDF_PATH)
#     blocks = extractor.extract_clean_text()
#     print(f"\nTotal extracted blocks: {len(blocks)}")
