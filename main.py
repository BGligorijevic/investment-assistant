import sys
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool
from langchain_community.tools.tavily_search import TavilySearchResults

# Import from local files in the simple structure
from financial_document_analyst import get_financial_document_answer
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

    # The agent now has two tools in its "toolbox"
    tools = [
        Tool(
            name="FinancialDocumentAnalyst",
            func=get_financial_document_answer,
            description="""
            Answers questions about financial data from a specific quarterly report PDF.
            Use this for any question that specifies a quarter and a year (e.g., 'q1 2025').
            The input MUST be a question for a SINGLE quarter and year.
            """,
        ),
        TavilySearchResults(max_results=3)
    ]

    # Pull the base ReAct prompt from the LangChain Hub
    # This prompt already contains the required placeholders like {tools} and {tool_names}
    prompt = hub.pull("hwchase17/react")

    # We can add our custom instructions to the prompt by using the .partial() method.
    # This pre-fills parts of the prompt template.
    # We will insert our rules into the 'tools' section of the prompt.
    # This new prompt is much more detailed about the logic of using the tools.
    prompt.template = """You are an expert AI financial assistant. You have two tools:

1. `FinancialDocumentAnalyst`: Use this tool to find information inside specific quarterly financial reports that are stored locally.
2. `tavily_search_results_json`: Use this tool to search the web for information you cannot find in the local documents.

**Your Logic Must Be:**
1.  **ALWAYS** start by using the `FinancialDocumentAnalyst` tool first.
2.  If the `FinancialDocumentAnalyst` tool returns an error or states that it cannot find a document, **ONLY THEN** should you use the `tavily_search_results_json` tool to find the answer on the web.
3.  If the user asks for a 'trend' over a year, you must call the `FinancialDocumentAnalyst` tool for each quarter of that year. If any of those calls fail, do not use the web search tool. Instead, inform the user which quarters you found data for and which you could not.
""" + prompt.template

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
