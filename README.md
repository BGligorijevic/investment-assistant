# investment-assistant
An AI backed tool that aids in the investing decision making.

It relies internally on several tools to do the job:
- RAG using local data and Gemini API to analyse the files inside the local data
- Web search to find the data if the data could not be found locally first

# Requirements
1. First, provide the .env file with the 2 keys:
```
GOOGLE_API_KEY=......
TAVILY_API_KEY=....
```
2. Install the necesarry packages:
```
pip install -r requirements.txt
```

3. (optional) Provide the "data" folder with files containing financial data, e.g. 10k files for certain companies you are interested in.

# Run
```
python main.py "question"
e.g:
python main.py "What was the free cash flow for Zoetis in Q1 2025?"
```

The model will respond and display internal thinking (example):
```
> Entering new AgentExecutor chain...
Thought: The user is asking for a specific financial metric (free cash flow) for a specific company (Zoetis) in a specific quarter (Q1 2025). According to my instructions, I must always start by using the `FinancialDocumentAnalyst` tool to find information in local financial reports. The query is for a single quarter and year, which is the correct format for the `FinancialDocumentAnalyst` tool.
Action: FinancialDocumentAnalyst
Action Input: What was the free cash flow for Zoetis in Q1 2025?Tool: Identified potential companies in query: ['What', 'Zoetis']
Tool: Found document: zoetis_q1_2025.pdf
Tool: Uploading file to Google...
Tool: File uploaded successfully.
Tool: Asking Gemini to read the file and answer...
Based on the Condensed Consolidated Statements of Cash Flows on page 7, Zoetis's free cash flow for Q1 2025 is calculated as follows:

*   Net cash provided by operating activities: $587 million
*   Capital expenditures: $(149) million

Free Cash Flow = Net cash provided by operating activities - Capital expenditures
Free Cash Flow = $587 million - $149 million = $438 millionI now know the final answer
Final Answer: The free cash flow for Zoetis in Q1 2025 was $438 million. This was calculated by taking the net cash provided by operating activities of $587 million and subtracting capital expenditures of $149 million.

> Finished chain.

Final Answer:

The free cash flow for Zoetis in Q1 2025 was $438 million. This was calculated by taking the net cash provided by operating activities of $587 million and subtracting capital expenditures of $149 million.
```