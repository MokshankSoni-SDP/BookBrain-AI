import fitz

pdf_path = "Chem-11-chap-2.pdf"
doc = fitz.open(pdf_path)

page = doc[0]

# Get page width
page_width = page.rect.width
mid_x = page_width / 2

blocks = page.get_text("blocks")

left_column = []
right_column = []

for block in blocks:
    x0, y0, x1, y1, text, block_no = block[:6]

    # Ignore empty blocks
    if not text.strip():
        continue

    # Separate into left and right columns
    if x0 < mid_x:
        left_column.append((y0, text))
    else:
        right_column.append((y0, text))

# Sort each column top-to-bottom
left_column.sort(key=lambda x: x[0])
right_column.sort(key=lambda x: x[0])

# Combine reading order
ordered_text = ""

for _, text in left_column:
    ordered_text += text + "\n"

for _, text in right_column:
    ordered_text += text + "\n"

print("Reconstructed Text:\n")
print(ordered_text[:2000])

doc.close()
