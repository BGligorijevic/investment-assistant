import json
from pathlib import Path

def create_structured_data_dataset():
    dataset_path = Path.cwd() / "subagents" / "structured_data_extractor" / "dataset.jsonl"
    dataset_path.parent.mkdir(exist_ok=True)

    context1 = """
        CONSOLIDATED STATEMENTS OF CASH FLOWS (Unaudited)
        (millions of dollars)
        For the Three Months Ended March 31, 2025
        Net income $ 530
        Depreciation and amortization 150
        Net cash provided by operating activities 587
        Capital expenditures (149)
        Net cash used in investing activities (140)
        Debt repayment (200)
        Net cash used in financing activities (250)
        """

    # Create multiple examples from a single context
    examples = [
        {"input": f"Context: {context1}\nQuestion: What is the net income?", "output": "530"},
        {"input": f"Context: {context1}\nQuestion: What is the depreciation and amortization?", "output": "150"},
        {"input": f"Context: {context1}\nQuestion: What is the net cash provided by operating activities?", "output": "587"},
        {"input": f"Context: {context1}\nQuestion: What are the capital expenditures?", "output": "-149"},
        {"input": f"Context: {context1}\nQuestion: What is the net cash used in investing activities?", "output": "-140"},
        {"input": f"Context: {context1}\nQuestion: What is the debt repayment?", "output": "-200"},
    ]

    print(f"Creating dataset with {len(examples)} examples at: {dataset_path}")

    with open(dataset_path, 'w') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')

    print("Dataset created successfully.")

if __name__ == "__main__":
    create_structured_data_dataset()