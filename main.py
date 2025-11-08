import sys
from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool
from langchain_community.tools.tavily_search import TavilySearchResults

from subagents.document_analyst.financial_document_analyst import get_financial_document_answer
from config import MANAGER_MODEL, GOOGLE_API_KEY, TAVILY_API_KEY # Ensure API keys are loaded

def run_assistant():
    # The query is now taken from the command line arguments
    if len(sys.argv) < 2:
        print("\nUsage: python main.py \"Your complex question here\"")
        print("Example: python main.py \"What was the EPS trend in 2025?\"")
        return

    query = " ".join(sys.argv[1:])
    print(f"Querying Agent: {query}\n")

    llm = ChatGoogleGenerativeAI(model=MANAGER_MODEL, temperature=0)

    # Start with the base tools that are always available
    base_tools = [
        Tool(
            name="FinancialDocumentAnalyst",
            func=get_financial_document_answer,
            description="""
            Answers questions about financial data from a specific quarterly report PDF.
            Use this for any question that specifies a quarter and a year (e.g., 'q1 2025').
            The input MUST be a question for a SINGLE quarter and year.
            """,
        ),
        TavilySearchResults(max_results=3),
    ]

    # --- Conditionally add the sentiment tool ---
    tools = base_tools
    sentiment_model_path = Path.cwd() / "models" / "sentiment_analyzer"
    prompt_instructions = """You are an expert AI financial assistant. You have two tools:

1. `FinancialDocumentAnalyst`: Use this tool to find information inside specific quarterly financial reports that are stored locally.
2. `tavily_search_results_json`: Use this tool to search the web for information you cannot find in the local documents.

**Your Logic Must Be:**
1.  **ALWAYS** start by using the `FinancialDocumentAnalyst` tool first for any query about financial data.
2.  If the `FinancialDocumentAnalyst` tool returns an error or states that it cannot find a document, **ONLY THEN** should you use the `tavily_search_results_json` tool to find the answer on the web.
3.  If the user asks for a 'trend' over a year, you must call the `FinancialDocumentAnalyst` tool for each quarter of that year. If any of those calls fail, do not use the web search tool. Instead, inform the user which quarters you found data for and which you could not.
"""

    if sentiment_model_path.exists():
        print("INFO: Fine-tuned sentiment model found. Adding SentimentAnalyzer tool.")
        from subagents.sentiment_analyzer.sentiment_analyzer import get_sentiment
        sentiment_tool = Tool(
                name="SentimentAnalyzer",
                func=get_sentiment,
                description="Use this tool to analyze the sentiment of a news headline or a short text. Input must be a single string of text."
            )
        tools.append(sentiment_tool)
        prompt_instructions = """You are an expert AI financial assistant. You have three tools:

1. `FinancialDocumentAnalyst`: Use this tool to find information inside specific quarterly financial reports that are stored locally.
2. `tavily_search_results_json`: Use this tool to search the web for information you cannot find in the local documents.
3. `SentimentAnalyzer`: Use this tool to find the sentiment of a news headline.

**Your Logic Must Be:**
1.  **ALWAYS** start by using the `FinancialDocumentAnalyst` tool first for any query about financial data.
2.  If the `FinancialDocumentAnalyst` tool returns an error or states that it cannot find a document, **ONLY THEN** should you use the `tavily_search_results_json` tool to find the answer on the web.
3.  If the user asks about news or sentiment (e.g., "What is the latest news about Google and is it positive?"), use `tavily_search_results_json` to find news articles, and then use the `SentimentAnalyzer` on each headline to determine the sentiment.
4.  If the user asks for a 'trend' over a year, you must call the `FinancialDocumentAnalyst` tool for each quarter of that year. If any of those calls fail, do not use the web search tool. Instead, inform the user which quarters you found data for and which you could not.
"""

    # Instead of pulling from the hub, we define the ReAct prompt template directly.
    # This removes the deprecation warning and makes the app self-contained.
    react_prompt_template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

CRITICAL RULE: The 'Action' and 'Action Input' must be two separate lines of plain text. DO NOT use JSON.

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

    # Combine your instructions with the ReAct template
    full_template = prompt_instructions + react_prompt_template
    prompt = PromptTemplate.from_template(full_template)

    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    try:
        response = agent_executor.invoke({"input": query})
        print("\nFinal Answer:\n")
        print(response["output"])
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    run_assistant()
