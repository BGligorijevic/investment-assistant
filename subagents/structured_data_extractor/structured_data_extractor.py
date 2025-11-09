from transformers import pipeline
import re
import json
from langchain_core.tools import Tool
from pathlib import Path

MODEL_PATH = Path.cwd() / "models" / "structured_data_extractor"

structured_data_pipeline = None

def get_structured_data_extractor_tool() -> Tool:
    return Tool(
        name="StructuredDataExtractor",
        func=get_structured_data,
        description="Parses the raw text of a financial cash flow statement to extract key metrics into a structured JSON format. The input should be a single string containing the raw text of the cash flow table."
    )

def get_structured_data(text: str) -> str:
    # This tool is now specialized for cash flow statements.
    # It contains the predefined questions to ask the fine-tuned model.
    questions = {
        "net_income": "What is the net income?",
        "depreciation_and_amortization": "What is the depreciation and amortization?",
        "net_cash_provided_by_operating_activities": "What is the net cash provided by operating activities?",
        "capital_expenditures": "What are the capital expenditures?",
        "net_cash_used_in_investing_activities": "What is the net cash used in investing activities?",
        "debt_repayment": "What is the debt repayment?"
    }

    global structured_data_pipeline
    if not MODEL_PATH.exists():
        return "Tool Error: The fine-tuned structured data model was not found. Please run the training script to enable this tool."

    if structured_data_pipeline is None:
        print("Tool: Loading structured data extraction model for the first time...")
        structured_data_pipeline = pipeline("text2text-generation", model=str(MODEL_PATH), tokenizer=str(MODEL_PATH), device=-1)

    results = {}
    for key, question in questions.items():
        prompt = f"Context: {text}\nQuestion: {question}"
        raw_answer = structured_data_pipeline(prompt, max_length=20)[0]['generated_text']
        
        # The model's output can be messy. We will use a regex to robustly
        # find the first number-like value in the answer. This handles cases
        # like "$ 530" or "150 Net cash...".
        match = re.search(r'(\d[\d,.]*)', raw_answer)
        
        cleaned_answer = raw_answer
        if match:
            num_str = match.group(1).replace(',', '')
            # Check if the number was in parentheses, which indicates a negative value
            if f"({num_str})" in raw_answer.replace(" ", ""):
                cleaned_answer = f"-{num_str}"
            else:
                cleaned_answer = num_str

        try:
            num_answer = float(cleaned_answer)
            if num_answer.is_integer():
                results[key] = int(num_answer)
            else:
                results[key] = num_answer
        except ValueError:
            results[key] = cleaned_answer

    return json.dumps(results)