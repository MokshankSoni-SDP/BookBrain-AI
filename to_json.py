import json
import config
from classify_blocks import classify_and_clean

def build_hierarchy(classified_blocks):
    root = {
        "chapter_title": "STRUCTURE OF ATOM",
        "sections": [],
        "exercises": []
    }
    
    current_section = None
    current_subsection = None
    in_exercises = False

    for block in classified_blocks:
        b_type = block["type"]
        b_text = block["text"]

        # 1. Handle Exercises
        if b_type == "EXERCISE" or in_exercises:
            if "EXERCISES" in b_text.upper(): in_exercises = True
            root["exercises"].append(b_text)
            continue

        # 2. Handle Headings
        if b_type == "HEADING":
            # Extra safety: Ensure the heading starts with the chapter number '2'
            if not b_text.startswith("2."):
                # If it's a fake heading (like a scientific constant), treat it as a paragraph
                b_type = "PARAGRAPH"
            else:
                section_id = b_text.split()[0]
                dots = section_id.count('.')

                if dots == 1: # Level 1: 2.1, 2.2
                    current_section = {
                        "id": section_id,
                        "title": b_text,
                        "content": [],
                        "subsections": []
                    }
                    root["sections"].append(current_section)
                    current_subsection = None 
                    continue # Move to next block

                elif dots == 2: # Level 2: 2.1.1, 2.1.2
                    current_subsection = {
                        "id": section_id,
                        "title": b_text,
                        "content": []
                    }
                    if current_section:
                        current_section["subsections"].append(current_subsection)
                    else:
                        # If 2.1.1 appears before 2.1, create a placeholder
                        current_section = {"id": "2.0", "title": "Intro", "content": [], "subsections": [current_subsection]}
                        root["sections"].append(current_section)
                    continue

        # 3. Handle Paragraphs, Figures, and Equations
        if current_subsection:
            current_subsection["content"].append(b_text)
        elif current_section:
            current_section["content"].append(b_text)
        else:
            if "preamble" not in root: root["preamble"] = []
            root["preamble"].append(b_text)

    return root

# ... (rest of save_structured_data() remains the same as before)

def save_structured_data():
    print(f"Starting JSON structuring for: {config.PDF_PATH}...")
    
    # Get classified data from Step 3
    raw_classified = classify_and_clean()
    
    # Build hierarchy
    final_json = build_hierarchy(raw_classified)
    
    # Save to file defined in config
    with open(config.OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(final_json, f, indent=4, ensure_ascii=False)
    
    print(f"Final Storage Complete: {config.OUTPUT_JSON}")

if __name__ == "__main__":
    save_structured_data()