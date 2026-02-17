import json
import config
from step3_classify_blocks import classify_and_clean


def build_hierarchy(classified_items):

    root = {
        "chapter_title": "SYSTEMS OF PARTICLES AND ROTATIONAL MOTION",
        "sections": [],
        "summary": [],
        "points_to_ponder": [],
        "exercises": []
    }

    curr_sec, curr_subsec = None, None
    mode = "theory"   # 🔥 NEW MODE TRACKING

    for item in classified_items:
        itype, ivalue = item["type"], item["value"].strip()

        upper_value = ivalue.upper()

        # --------------------------------------------------
        # 🔥 MODE SWITCHING
        # --------------------------------------------------
        if upper_value == "SUMMARY":
            mode = "summary"
            continue

        if upper_value == "POINTS TO PONDER":
            mode = "points_to_ponder"
            continue

        if upper_value == "EXERCISES":
            mode = "exercises"
            continue

        # --------------------------------------------------
        # 🔥 HANDLE END-SECTIONS
        # --------------------------------------------------
        if mode == "summary":
            root["summary"].append(ivalue)
            continue

        if mode == "points_to_ponder":
            root["points_to_ponder"].append(ivalue)
            continue

        if mode == "exercises":
            root["exercises"].append(ivalue)
            continue

        # --------------------------------------------------
        # NORMAL THEORY MODE
        # --------------------------------------------------
        if itype == "HEADING":
            id_tag = ivalue.split()[0]
            level = id_tag.count('.')

            if level == 1:
                curr_sec = {
                    "id": id_tag,
                    "title": ivalue,
                    "content": [],
                    "subsections": []
                }
                root["sections"].append(curr_sec)
                curr_subsec = None

            elif level == 2:
                curr_subsec = {
                    "id": id_tag,
                    "title": ivalue,
                    "content": []
                }
                if curr_sec:
                    curr_sec["subsections"].append(curr_subsec)

            continue

        # Content placement
        if curr_subsec:
            curr_subsec["content"].append(ivalue)
        elif curr_sec:
            curr_sec["content"].append(ivalue)
        else:
            if "preamble" not in root:
                root["preamble"] = []
            root["preamble"].append(ivalue)

    print(f"[DEBUG] Sections found: {len(root['sections'])}")
    print(f"[DEBUG] Summary items: {len(root['summary'])}")
    print(f"[DEBUG] Points to ponder items: {len(root['points_to_ponder'])}")
    print(f"[DEBUG] Exercises items: {len(root['exercises'])}")

    return root


def save_structure(structure, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(structure, f, indent=4, ensure_ascii=False)

    print(f"Saved structure to: {output_path}")


if __name__ == "__main__":
    raw_data = classify_and_clean()
    structured_json = build_hierarchy(raw_data)
    save_structure(structured_json, config.OUTPUT_JSON)
