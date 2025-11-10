import sys
from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import Tool
from subagents.document_analyst.financial_document_analyst import get_financial_document_tool
from config import MANAGER_MODEL
from subagents.web_search.web_search import get_web_search_tool

def run_assistant():
    if len(sys.argv) < 2:
        print("\nUsage: python main.py \"Your complex question here\"")
        print("Example: python main.py \"What was the EPS trend for Alphabet in 2025?\"")
        return

    query = " ".join(sys.argv[1:])
    print(f"Querying Agent: {query}\n")

    llm = ChatGoogleGenerativeAI(model=MANAGER_MODEL, temperature=0)

    # Start with the base tools that are always available
    base_tools = [
        get_financial_document_tool(),
        get_web_search_tool(max_results=3),
    ]

    tools = base_tools

    # Conditionally add the sentiment tool
    sentiment_model_path = Path.cwd() / "models" / "sentiment_analyzer"
    if sentiment_model_path.exists():
        print("INFO: Fine-tuned sentiment model found. Adding SentimentAnalyzer tool.")
        from subagents.sentiment_analyzer.sentiment_analyzer import get_sentiment_tool
        tools.append(get_sentiment_tool())

    # Conditionally add the structured data extractor tool
    structured_model_path = Path.cwd() / "models" / "structured_data_extractor"
    if structured_model_path.exists():
        print("INFO: Fine-tuned structured data model found. Adding StructuredDataExtractor tool.")
        from subagents.structured_data_extractor.structured_data_extractor import get_structured_data_extractor_tool
        tools.append(get_structured_data_extractor_tool())

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             """You are an expert AI financial assistant.
Your primary goal is to answer financial questions using local documents first.
- For questions about specific financial data (e.g., from a specific quarter), ALWAYS use the `FinancialDocumentAnalyst` tool.
- If the user asks to 'parse a cash flow statement', 'extract a cash flow table', or wants 'JSON from a cash flow statement', you must follow these two steps:
  1. First, use the `FinancialDocumentAnalyst` tool to get the raw text of the table from the specified document. Your input to this tool should be a question like "Extract the text of the consolidated statements of cash flows from [document name]".
  2. Second, take the raw text output from the `FinancialDocumentAnalyst` and pass it directly as the input to the `StructuredDataExtractor` tool.
- If the `FinancialDocumentAnalyst` tool cannot find the information, then use the web search tool.
- For questions about news or sentiment, use the web search tool to find articles, then use the `SentimentAnalyzer` on the headlines.
- To analyze a 'trend' over a year, you must call the `FinancialDocumentAnalyst` tool for each quarter of that year. If any calls fail, report what you found and what was missing. Do not use the web search tool to fill in missing quarters for a trend analysis.
"""),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

    try:
        response = agent_executor.invoke({"input": query})
        print("\nFinal Answer:\n")
        print(response["output"])
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    run_assistant()