import pdfplumber

pdf_file = 'arkansas_police_1033_proposal.pdf'

print("--- Checking PDF Contents ---")
with pdfplumber.open(pdf_file) as pdf:
    print(f"Total pages found: {len(pdf.pages)}")
    
    # Let's extract and print the raw text from Page 1 to see how it looks
    first_page_text = pdf.pages[0].extract_text()
    
    if first_page_text:
        print("\n✅ Text found on Page 1! Here are the first 500 characters:\n")
        print("-" * 50)
        print(first_page_text[:500])
        print("-" * 50)
    else:
        print("\n❌ The text on Page 1 is completely empty. The PDF might be a scanned image.")