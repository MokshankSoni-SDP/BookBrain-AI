# config.py

PDF_PATH = "Physics-11 1-92-126.pdf"
OUTPUT_JSON = "physics_structure.json"

IMAGE_DIR = "./extract_images"

RULES = {
    # Added \s+ to ensure we match "6.1 INTRODUCTION" correctly
    "HEADING_PATTERN": r'^\d+\.\d+(\.\d+)?\s+', 
    "FIGURE_PATTERN": r'^(?:Fig\.|Figure)\s*\d+\.\d+', # Stricter start-of-line match
    "EQUATION_PATTERN": r'[\(\[](\d+\.\d+)[\)\]]',
    "EXERCISE_KEYWORDS": ["EXERCISES", "Problem", "Question"]
}

# Adjusted to ~240pt on a 600pt page to separate the sidebar
COLUMN_GAP_THRESHOLD = 0.4
CAPTION_LOOK_AHEAD = 2