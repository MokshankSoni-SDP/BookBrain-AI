import fitz
import re
import os
import config


class NCERTFigureExtractor:

    def __init__(self, pdf_path, output_folder="extracted_images"):
        self.doc = fitz.open(pdf_path)
        self.output_folder = output_folder

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        # Matches:
        # Fig. 6.4
        # Fig. 6.5(a)
        # Fig. 6.6(b)
        self.fig_pattern = re.compile(r"Fig\.\s*(\d+)\.(\d+)(\([a-z]\))?", re.IGNORECASE)

    # ---------------------------------------------------
    # Convert match to clean filename
    # ---------------------------------------------------
    def build_filename(self, match):
        chapter = match.group(1)
        fig = match.group(2)
        sub = match.group(3)

        if sub:
            sub = sub.replace("(", "").replace(")", "")
            return f"fig_{chapter}_{fig}_{sub}.png"
        else:
            return f"fig_{chapter}_{fig}.png"

    # ---------------------------------------------------
    # Extract figures using caption anchor
    # ---------------------------------------------------
    def extract_figures(self):

        total_figures = 0

        for page_number, page in enumerate(self.doc):

            page_dict = page.get_text("dict")

            for block in page_dict["blocks"]:
                if block["type"] != 0:
                    continue

                for line in block["lines"]:

                        # Reconstruct full line text
                        full_line = ""
                        for span in line["spans"]:
                            full_line += span["text"]
                    
                        full_line = full_line.strip()
                    
                        match = self.fig_pattern.search(full_line)
                        if not match:
                            continue
                        
                        # Use first span bbox for caption position
                        x0, y0, x1, y1 = line["spans"][0]["bbox"]
                    
                        filename = self.build_filename(match)
                    
                        image_top = max(0, y0 - 450)
                        image_bottom = y0 - 5
                    
                        image_rect = fitz.Rect(
                            0,
                            image_top,
                            page.rect.width,
                            image_bottom
                        )
                    
                        pix = page.get_pixmap(clip=image_rect, dpi=300)
                        save_path = os.path.join(self.output_folder, filename)
                        pix.save(save_path)
                    
                        print(f"Saved: {filename} (Page {page_number + 1})")


                        # Caption found
                        filename = self.build_filename(match)

                        # Caption bounding box
                        x0, y0, x1, y1 = span["bbox"]

                        # -----------------------------------------
                        # Define image region ABOVE caption
                        # -----------------------------------------

                        image_top = max(0, y0 - 450)  # adjust height if needed
                        image_bottom = y0 - 5

                        image_rect = fitz.Rect(
                            0,
                            image_top,
                            page.rect.width,
                            image_bottom
                        )

                        try:
                            pix = page.get_pixmap(clip=image_rect, dpi=300)
                            save_path = os.path.join(self.output_folder, filename)
                            pix.save(save_path)

                            print(f"Saved: {filename} (Page {page_number + 1})")
                            total_figures += 1

                        except Exception as e:
                            print(f"Failed extracting {filename}: {e}")

        print(f"\nTotal Figures Extracted: {total_figures}")


# ---------------------------------------------------
# Run
# ---------------------------------------------------
if __name__ == "__main__":

    extractor = NCERTFigureExtractor(config.PDF_PATH)
    extractor.extract_figures()
