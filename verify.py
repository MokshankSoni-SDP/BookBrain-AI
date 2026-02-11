import fitz  # This is PyMuPDF
import sys

def verify():
    print(f"Python version: {sys.version}")
    print(f"PyMuPDF version: {fitz.version}")
    
    try:
        # Replace with your actual filename if different
        doc = fitz.open("Chem-11-chap-2.pdf") 
        print(f"\nSuccessfully opened: {doc.name}")
        print(f"Total Pages: {doc.page_count}")
        
        # Test reading the first few lines of the first page
        page = doc[0]
        text = page.get_text()
        print("\nFirst 100 characters of Page 1:")
        print("-" * 30)
        print(text[:100])
        print("-" * 30)
        
        doc.close()
        print("\nVerification Successful! We can now move to Step 3.")
    except Exception as e:
        print(f"\nError: Could not open the PDF. Ensure the file is in the same folder. \nDetails: {e}")

if __name__ == "__main__":
    verify()