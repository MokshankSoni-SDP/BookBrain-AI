
import os
import shutil
import uuid
from ingest import ingest_data
from step3_classify_blocks import classify_and_clean
from step4_to_json import build_hierarchy, save_structure

def run_pdf_pipeline(pdf_path, output_dir="./processed_data", client=None):
    """
    Runs the full pipeline:
    1. Extract blocks & images (step 3)
    2. Build JSON hierarchy (step 4)
    3. Ingest into Qdrant
    """
    
    # Create unique output directory for this run to avoid collisions
    run_id = str(uuid.uuid4())[:8]
    run_dir = os.path.join(output_dir, run_id)
    images_dir = os.path.join(run_dir, "images")
    json_path = os.path.join(run_dir, "structure.json")
    
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    
    print(f"[Pipeline] Starting processing for {pdf_path}")
    print(f"[Pipeline] Output directory: {run_dir}")

    # Step 1 & 2: Classify blocks and extract images
    print("[Pipeline] Classifying blocks and extracting images...")
    # classify_and_clean now accepts pdf_path and image_output_dir
    classified_items = classify_and_clean(pdf_path=pdf_path, image_output_dir=images_dir)
    
    # Step 3: Build JSON
    print("[Pipeline] Building hierarchy...")
    hierarchy = build_hierarchy(classified_items)
    save_structure(hierarchy, json_path)
    
    # Step 4: Ingest
    print("[Pipeline] Ingesting into Qdrant...")
    # We pass the JSON path to ingest_data
    ingest_data(json_path, client=client)
    
    print(f"[Pipeline] Complete! Data ingested from {pdf_path}")
    return run_dir, json_path, images_dir

if __name__ == "__main__":
    # Test run
    # run_pdf_pipeline("Physics-11 1-92-126.pdf")
    pass
