# config.py

PDF_PATH = "Physics-11 1-92-126.pdf"
OUTPUT_JSON = "physics_structure.json"

#IMAGE_DIR = "./extract_images"
IMAGE_DIR = "processed_data/knowledge_base/extracted_images"

RULES = {
    "HEADING_PATTERN": r'^\d+\.\d+(\.\d+)?\s+',
    "FIGURE_PATTERN": r'(?:Fig\.|Figure)\s*(\d+\.\d+)',
    "EQUATION_PATTERN": r'[\(\[](\d+\.\d+)[\)\]]',
    "CHAPTER_PATTERN_12": r'^Chapter\s+[A-Za-z]+$',

    # Added semantic end-of-chapter triggers
    "END_SECTION_KEYWORDS": [
        "SUMMARY",
        "POINTS TO PONDER",
        "EXERCISES"
    ],

    "EXERCISE_KEYWORDS": [
        "EXERCISES",
        "Problem",
        "Question"
    ]
}

COLUMN_GAP_THRESHOLD = 0.4
CAPTION_LOOK_AHEAD = 2
