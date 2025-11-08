import re
from pathlib import Path
import google.generativeai as genai

# Import from local config file
from config import VISUAL_ANALYST_MODEL

# Point to our data folder - this assumes a `data` folder at the project root
DATA_DIR = Path.cwd() / "data"

def get_financial_document_answer(query: str) -> str:
    """
    Finds the right PDF from the /data folder, uploads it to Gemini, and asks
    a question about it. The query MUST contain a quarter and a year.
    """
    
    # 1. Parse the query to find the file
    quarter_match = re.search(r"q(\d)", query.lower())
    year_match = re.search(r"(20\d\d)", query.lower())

    if not (quarter_match and year_match):
        return "Tool Error: A specific quarter and year are required (e.g., 'q2 2025')."

    quarter = f"q{quarter_match.group(1)}"
    year = year_match.group(1)
    
    # --- NEW LOGIC ---
    # Extract potential company names from the query.
    # We'll split the query and look for capitalized words that aren't the quarter.
    potential_companies = [word for word in query.split() if word.istitle() and not re.match(r"Q\d", word, re.IGNORECASE)]
    if not potential_companies:
        return "Tool Error: No company name could be identified in the query."
    
    print(f"Tool: Identified potential companies in query: {potential_companies}")

    # Find the matching PDF in the data/ folder
    if not DATA_DIR.exists():
        return f"Tool Error: The data directory was not found at {DATA_DIR}"
        
    target_filename = None
    for f in DATA_DIR.glob("*.pdf"):
        # Check for company, quarter, AND year in the filename
        if quarter in f.name.lower() and year in f.name.lower() and any(company.lower() in f.name.lower() for company in potential_companies):
            target_filename = f
            break
            
    if not target_filename:
        return f"Tool Error: I couldn't find a PDF in my data folder for {quarter} {year}."

    print(f"Tool: Found document: {target_filename.name}")
    print("Tool: Uploading file to Google...")
    
    file_to_upload = genai.upload_file(path=target_filename, display_name=f"{quarter} {year} Report")
    print("Tool: File uploaded successfully.")

    model = genai.GenerativeModel(model_name=VISUAL_ANALYST_MODEL)   
     
    prompt_parts = [
        file_to_upload,
        "You are an expert financial analyst.",
        "The user has provided you with a financial report.",
        "Please answer the user's question based *only* on the contents of this file.",
        f"\nUser Question: {query}",
    ]

    print("Tool: Asking Gemini to read the file and answer...")
    try:
        response = model.generate_content(prompt_parts)
        genai.delete_file(file_to_upload.name)
        return response.text
    except Exception as e:
        genai.delete_file(file_to_upload.name)
        return f"Tool Error: An error occurred while processing the file: {e}"