import re
import json
from step1_column_extractor import ProductionColumnExtractor
import config


class NCERTStructureBuilder:

    def __init__(self):
        self.chapter = {
            "chapter_number": 6,
            "chapter_title": "",
            "sections": [],
            "summary": [],
            "points_to_ponder": [],
            "exercises": []
        }

        self.current_section = None
        self.current_subsection = None

        self.in_summary = False
        self.in_points = False
        self.in_exercises = False

        self.started_real_content = False

    # ----------------------------
    # Detection Helpers
    # ----------------------------

    def is_real_section_start(self, text):
        # Detect 6.1 INTRODUCTION (uppercase title)
        return bool(re.match(r"^6\.1\s+[A-Z]{3,}", text.strip()))

    def is_section(self, text):
        return bool(re.match(r"^6\.\d+\s+[A-Z]", text.strip()))

    def is_subsection(self, text):
        return bool(re.match(r"^6\.\d+\.\d+\s+", text.strip()))

    def is_summary(self, text):
        return text.strip().upper() == "SUMMARY"

    def is_points(self, text):
        return text.strip().upper() == "POINTS TO PONDER"

    def is_exercises(self, text):
        return text.strip().upper() == "EXERCISES"

    # ----------------------------
    # Main Builder
    # ----------------------------

    def build(self, blocks):

        for block in blocks:

            if block["type"] != "text":
                continue

            text = block["content"].strip()

            if not text:
                continue

            # -----------------------------
            # Capture chapter title
            # -----------------------------
            if (
                self.chapter["chapter_title"] == ""
                and text.isupper()
                and "SYSTEMS OF PARTICLES" in text
            ):
                self.chapter["chapter_title"] = text
                continue

            # -----------------------------
            # Skip TOC until real 6.1 starts
            # -----------------------------
            if not self.started_real_content:
                if self.is_real_section_start(text):
                    self.started_real_content = True
                else:
                    continue

            # -----------------------------
            # Detect Summary / Points / Exercises
            # -----------------------------
            if self.is_summary(text):
                self.in_summary = True
                self.in_points = False
                self.in_exercises = False
                continue

            if self.is_points(text):
                self.in_points = True
                self.in_summary = False
                self.in_exercises = False
                continue

            if self.is_exercises(text):
                self.in_exercises = True
                self.in_summary = False
                self.in_points = False
                continue

            if self.in_summary:
                self.chapter["summary"].append(text)
                continue

            if self.in_points:
                self.chapter["points_to_ponder"].append(text)
                continue

            if self.in_exercises:
                self.chapter["exercises"].append(text)
                continue

            # -----------------------------
            # Section
            # -----------------------------
            if self.is_section(text):
                section_number = text.split()[0]
                section_title = text[len(section_number):].strip()

                self.current_section = {
                    "section_number": section_number,
                    "section_title": section_title,
                    "content": [],
                    "subsections": []
                }

                self.chapter["sections"].append(self.current_section)
                self.current_subsection = None
                continue

            # -----------------------------
            # Subsection
            # -----------------------------
            if self.is_subsection(text):
                subsection_number = text.split()[0]
                subsection_title = text[len(subsection_number):].strip()

                self.current_subsection = {
                    "subsection_number": subsection_number,
                    "subsection_title": subsection_title,
                    "content": []
                }

                if self.current_section:
                    self.current_section["subsections"].append(self.current_subsection)

                continue

            # -----------------------------
            # Normal Content
            # -----------------------------
            if self.current_subsection:
                self.current_subsection["content"].append(text)
            elif self.current_section:
                self.current_section["content"].append(text)

        return self.chapter


# ----------------------------
# Runner
# ----------------------------
if __name__ == "__main__":

    extractor = ProductionColumnExtractor(config.PDF_PATH)
    blocks = extractor.extract_clean_text()

    builder = NCERTStructureBuilder()
    chapter_json = builder.build(blocks)

    with open("chapter_structure.json", "w", encoding="utf-8") as f:
        json.dump(chapter_json, f, indent=4, ensure_ascii=False)

    print("JSON file generated successfully.")
