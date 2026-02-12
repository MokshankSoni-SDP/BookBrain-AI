import re
import json
import config
# Ensure this matches your actual extractor filename
from step1_column_extractor import ImprovedColumnExtractor 

class EnhancedStructureBuilder:
    """Builds hierarchical JSON structure from NCERT Physics textbook content"""

    def __init__(self):
        self.chapter = {
            "chapter_number": "",
            "chapter_title": "",
            "sections": []
        }
        self.current_section = None
        self.current_subsection = None

    def is_chapter_header(self, text):
        """Detects 'CHAPTER SIX' or similar"""
        return bool(re.match(r'^CHAPTER\s+[A-Z\d]+', text.strip(), re.I))

    def is_chapter_title(self, text):
        """Detects main title like 'SYSTEMS OF PARTICLES...'"""
        # NCERT titles are usually all caps and multi-word
        return (text.isupper() and len(text) > 15 and "MOTION" in text or "SYSTEMS" in text)

    def is_section(self, text):
        """Detect section: e.g., '6.1 INTRODUCTION'"""
        # Corrected regex: \d+ instead of d+
        return bool(re.match(r'^\d+\.\d+\s+[A-Z]', text.strip()))

    def is_subsection(self, text):
        """Detect subsection: e.g., '6.1.1 What kind...'"""
        return bool(re.match(r'^\d+\.\d+\.\d+\s+', text.strip()))

    def extract_section_info(self, text):
        """Separates '6.1' and 'INTRODUCTION'"""
        match = re.match(r'^(\d+\.\d+)\s+(.+)$', text.strip())
        if match:
            return match.group(1), match.group(2)
        return None, text.strip()

    def extract_subsection_info(self, text):
        """Separates '6.1.1' and Title"""
        match = re.match(r'^(\d+\.\d+\.\d+)\s+(.+)$', text.strip())
        if match:
            return match.group(1), match.group(2)
        return None, text.strip()

    def process_content_block(self, block):
        """Categorizes block as structural or content"""
        if block['type'] == 'text':
            content = block['content'].strip()
            if not content:
                return None
            
            # 1. Detect Chapter Number
            if self.is_chapter_header(content) and not self.chapter['chapter_number']:
                self.chapter['chapter_number'] = content.replace("CHAPTER", "").strip()
                return None

            # 2. Detect Chapter Title
            if self.is_chapter_title(content) and not self.chapter['chapter_title']:
                self.chapter['chapter_title'] = content
                return None

            # 3. Detect Sections
            if self.is_section(content):
                num, title = self.extract_section_info(content)
                self.current_section = {
                    'section_number': num,
                    'section_title': title,
                    'content': [],
                    'subsections': []
                }
                self.chapter['sections'].append(self.current_section)
                self.current_subsection = None # Reset subsection on new section
                return None
            
            # 4. Detect Subsections
            if self.is_subsection(content):
                num, title = self.extract_subsection_info(content)
                if self.current_section is not None:
                    self.current_subsection = {
                        'subsection_number': num,
                        'subsection_title': title,
                        'content': []
                    }
                    self.current_section['subsections'].append(self.current_subsection)
                    return None
            
            # 5. Regular Paragraph/Formula
            return {
                'type': 'formula' if block.get('is_math') else 'paragraph',
                'content': content
            }
        
        elif block['type'] == 'image':
            return {
                'type': 'image',
                'filename': block.get('filename', 'unknown.png'),
                'caption': block.get('caption', ''),
                'path': block.get('path', '')
            }
        
        return None

    def add_to_structure(self, item):
        """Helper to place content in the right hierarchy"""
        if not item: return

        if self.current_subsection:
            self.current_subsection['content'].append(item)
        elif self.current_section:
            self.current_section['content'].append(item)

    def build(self, all_blocks):
        """Processes flat list of blocks into hierarchical JSON"""
        for block in all_blocks:
            if block['type'] == 'page_break':
                continue
            
            processed_item = self.process_content_block(block)
            self.add_to_structure(processed_item)
            
        return self.chapter


if __name__ == "__main__":
    print("=" * 60)
    print("NCERT PHYSICS STRUCTURE BUILDER")
    print("=" * 60)
    
    # 1. Extraction
    print("\n[1/2] Extracting PDF content...")
    extractor = EnhancedStructureBuilder(config.PDF_PATH)
    blocks = extractor.extract_all_pages()
    extractor.close()
    
    # 2. Structuring
    print(f"\n[2/2] Building hierarchy from {len(blocks)} blocks...")
    builder = EnhancedStructureBuilder()
    chapter_json = builder.build(blocks)
    
    # 3. Output
    output_path = config.JSON_OUTPUT_FILE
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chapter_json, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 60)
    print("SUCCESS!")
    print(f"✓ Chapter: {chapter_json['chapter_number']} - {chapter_json['chapter_title']}")
    print(f"✓ Sections Detected: {len(chapter_json['sections'])}")
    print(f"✓ File Saved: {output_path}")

# import re
# import json
# from step1_column_extractor import ProductionColumnExtractor
# import config


# class NCERTStructureBuilder:

#     def __init__(self):
#         self.chapter = {
#             "chapter_number": 6,
#             "chapter_title": "",
#             "sections": [],
#             "summary": [],
#             "points_to_ponder": [],
#             "exercises": []
#         }

#         self.current_section = None
#         self.current_subsection = None

#         self.in_summary = False
#         self.in_points = False
#         self.in_exercises = False

#         self.started_real_content = False

#     # ----------------------------
#     # Detection Helpers
#     # ----------------------------

#     def is_real_section_start(self, text):
#         # Detect 6.1 INTRODUCTION (uppercase title)
#         return bool(re.match(r"^6\.1\s+[A-Z]{3,}", text.strip()))

#     def is_section(self, text):
#         return bool(re.match(r"^6\.\d+\s+[A-Z]", text.strip()))

#     def is_subsection(self, text):
#         return bool(re.match(r"^6\.\d+\.\d+\s+", text.strip()))

#     def is_summary(self, text):
#         return text.strip().upper() == "SUMMARY"

#     def is_points(self, text):
#         return text.strip().upper() == "POINTS TO PONDER"

#     def is_exercises(self, text):
#         return text.strip().upper() == "EXERCISES"

#     # ----------------------------
#     # Main Builder
#     # ----------------------------

#     def build(self, blocks):

#         for block in blocks:

#             if block["type"] != "text":
#                 continue

#             text = block["content"].strip()

#             if not text:
#                 continue

#             # -----------------------------
#             # Capture chapter title
#             # -----------------------------
#             if (
#                 self.chapter["chapter_title"] == ""
#                 and text.isupper()
#                 and "SYSTEMS OF PARTICLES" in text
#             ):
#                 self.chapter["chapter_title"] = text
#                 continue

#             # -----------------------------
#             # Skip TOC until real 6.1 starts
#             # -----------------------------
#             if not self.started_real_content:
#                 if self.is_real_section_start(text):
#                     self.started_real_content = True
#                 else:
#                     continue

#             # -----------------------------
#             # Detect Summary / Points / Exercises
#             # -----------------------------
#             if self.is_summary(text):
#                 self.in_summary = True
#                 self.in_points = False
#                 self.in_exercises = False
#                 continue

#             if self.is_points(text):
#                 self.in_points = True
#                 self.in_summary = False
#                 self.in_exercises = False
#                 continue

#             if self.is_exercises(text):
#                 self.in_exercises = True
#                 self.in_summary = False
#                 self.in_points = False
#                 continue

#             if self.in_summary:
#                 self.chapter["summary"].append(text)
#                 continue

#             if self.in_points:
#                 self.chapter["points_to_ponder"].append(text)
#                 continue

#             if self.in_exercises:
#                 self.chapter["exercises"].append(text)
#                 continue

#             # -----------------------------
#             # Section
#             # -----------------------------
#             if self.is_section(text):
#                 section_number = text.split()[0]
#                 section_title = text[len(section_number):].strip()

#                 self.current_section = {
#                     "section_number": section_number,
#                     "section_title": section_title,
#                     "content": [],
#                     "subsections": []
#                 }

#                 self.chapter["sections"].append(self.current_section)
#                 self.current_subsection = None
#                 continue

#             # -----------------------------
#             # Subsection
#             # -----------------------------
#             if self.is_subsection(text):
#                 subsection_number = text.split()[0]
#                 subsection_title = text[len(subsection_number):].strip()

#                 self.current_subsection = {
#                     "subsection_number": subsection_number,
#                     "subsection_title": subsection_title,
#                     "content": []
#                 }

#                 if self.current_section:
#                     self.current_section["subsections"].append(self.current_subsection)

#                 continue

#             # -----------------------------
#             # Normal Content
#             # -----------------------------
#             if self.current_subsection:
#                 self.current_subsection["content"].append(text)
#             elif self.current_section:
#                 self.current_section["content"].append(text)

#         return self.chapter


# # ----------------------------
# # Runner
# # ----------------------------
# if __name__ == "__main__":

#     extractor = ProductionColumnExtractor(config.PDF_PATH)
#     blocks = extractor.extract_clean_text()

#     builder = NCERTStructureBuilder()
#     chapter_json = builder.build(blocks)

#     with open("chapter_structure.json", "w", encoding="utf-8") as f:
#         json.dump(chapter_json, f, indent=4, ensure_ascii=False)

#     print("JSON file generated successfully.")
