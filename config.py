import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file. Please create a .env file and add it.")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY not found in .env file. Please create a .env file and add it.")

genai.configure(api_key=GOOGLE_API_KEY)

MANAGER_MODEL = "gemini-2.5-pro"            # The official API identifier for the latest Pro model.
VISUAL_ANALYST_MODEL = "gemini-2.5-flash"   # The official API identifier for the latest Flash model.