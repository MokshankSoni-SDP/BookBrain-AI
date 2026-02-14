import json
import config
from step3_classify_blocks import classify_and_clean

def build_hierarchy(classified_items):
    root = {
        "chapter_title": "SYSTEMS OF PARTICLES AND ROTATIONAL MOTION", 
        "sections": [], 
        "exercises": []
    }
    curr_sec, curr_subsec = None, None
    in_exercises = False

    for item in classified_items:
        itype, ivalue = item["type"], item["value"]

        # 1. Exercises handling
        if itype == "EXERCISE" or in_exercises:
            if "EXERCISES" in ivalue.upper(): in_exercises = True
            root["exercises"].append(ivalue)
            continue

        # 2. Section/Subsection logic
        if itype == "HEADING":
            id_tag = ivalue.split()[0]
            level = id_tag.count('.')
            
            if level == 1: # e.g., 6.1
                curr_sec = {"id": id_tag, "title": ivalue, "content": [], "subsections": []}
                root["sections"].append(curr_sec)
                curr_subsec = None
            elif level == 2: # e.g., 6.1.1
                curr_subsec = {"id": id_tag, "title": ivalue, "content": []}
                if curr_sec:
                    curr_sec["subsections"].append(curr_subsec)
            continue

        # 3. Content Placement (Pure strings in a list)
        if curr_subsec:
            curr_subsec["content"].append(ivalue)
        elif curr_sec:
            curr_sec["content"].append(ivalue)
        else:
            if "preamble" not in root: root["preamble"] = []
            root["preamble"].append(ivalue)

    print(f"[DEBUG] build_hierarchy: Found {len(root['sections'])} sections.")
    for sec in root["sections"]:
        print(f"    - Section {sec['id']}: {sec['title']} (Content: {len(sec['content'])} blocks, Subsections: {len(sec['subsections'])})")

    return root

def save_structure(structure, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(structure, f, indent=4, ensure_ascii=False)
    print(f"Saved structure to: {output_path}")

if __name__ == "__main__":
    raw_data = classify_and_clean() # uses config defaults
    structured_json = build_hierarchy(raw_data)
    save_structure(structured_json, config.OUTPUT_JSON)
    print(f"Success! Reverted to older string-list pattern in: {config.OUTPUT_JSON}")