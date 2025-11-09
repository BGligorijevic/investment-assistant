# Investment Assistant
Investment Assistant is a proof-of-concept AI agent designed to aid in investment research and decision-making.

It relies internally on several tools to do the job:
*   **Document Analyzer (RAG):** A Retrieval-Augmented Generation system using the Gemini API to analyze financial reports from a local, proprietary knowledge base.
*   **Structured Data Extractor:** A tool that uses a locally-run, fine-tuned model to parse unstructured text from a financial table (e.g., a cash flow statement) into a structured, machine-readable JSON object. This enables data-entry automation.
*   **Web Search:** A tool to search the web for real-time information and news.
*   **Fine-Tuned Sentiment Analyzer:** A custom classification locally-run model, fine-tuned on financial data to provide domain-specific sentiment scores for news headlines.

## Requirements
1. Decent hardware
2. Provide the .env file with the 2 keys:
```
GOOGLE_API_KEY=......
TAVILY_API_KEY=....
```
3. Install the necesarry packages:
```
pip install -r requirements.txt
```

4. (optional) Provide the "data" folder with files containing financial data under `subagents/document_analyst`, e.g. 10k files for certain companies you are interested in.
This is meant to hold proprietary data about certain companies not found on the internet, from internal sources, for example.

5. (optional) If sentiment analysis is required, fine-tune the sentiment analysis subagent's model (one-time action):
```
python subagents/sentiment_analyzer/train_sentiment_model.py
```
The training script is configured to run on the CPU by default (`no_cuda=True`). This is to ensure compatibility and avoid potential runtime errors with Apple Silicon (MPS) GPUs. If you are running on a machine with a compatible NVIDIA GPU and have CUDA installed, you can remove the `no_cuda=True` parameter from the `TrainingArguments` to enable GPU-accelerated training.

6. (optional) If structured data extraction is required, prepare the dataset and fine-tune the structured data extraction subagent's model (one-time action):
```
python subagents/structured_data_extractor/create_dataset.py
python subagents/structured_data_extractor/train_structured_data_extractor.py
```

## Running and examples
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

The model is also capable of analysing sentiment about a company.
E.g. asking an agent:
```
python main.py "What is the latest news about Google and is it positive?"
```
Would result in:

```> Entering new AgentExecutor chain...
Thought: The user is asking for the latest news about Google and whether it is positive. To answer this, I need to first find the latest news and then analyze the sentiment of the headlines.

1.  **Find the latest news:** I will use the `tavily_search_results_json` tool to search for "latest news about Google".
2.  **Analyze sentiment:** Once I have the news headlines, I will use the `SentimentAnalyzer` tool on each headline to determine if the sentiment is positive, negative, or neutral.
3.  **Synthesize the answer:** I will then combine the news and the sentiment analysis to provide a comprehensive answer to the user.
Action: tavily_search_results_json
Action Input: "latest news about Google"[{'url': 'https://www.cbsnews.com/tag/google/', 'content': "Watch CBS News\n\n## Google\n\n \n\n#### \n\n#### Update on Google's quantum computer\n\nGoogle recently said the quantum computer it's developing can run software 13,000 times as fast as a traditional super computer, according to reporting from the New York Times. New York Times technology reporter Cade Metz joins CBS News to discuss.\n\n \n\n#### OpenAI challenges Google Chrome [...] #### Google to spend $1 billion on AI education and job training in U.S.\n\nGoogle is partnering with colleges to give students free access to its artificial intelligence tools.\n\nScott Pelley learns about AI at DeepMind \n\n#### Google DeepMind CEO demonstrates Genie 2\n\nGoogle DeepMind CEO Demis Hassabis showed 60 Minutes Genie 2, an AI model that generates 3D interactive environments, which could be used to train robots in the not-so-distant future.\n\nDemis Hassabis [...] #### Google, OpenAI, Spotify and other platforms hit by widespread outage\n\nGoogle, OpenAI and Spotify were down Thursday after a widespread tech outage.\n\nSenate Majority Leader John Thune of South Dakota speaks to reporters at the Capitol on Nov. 7, 2025. \n\n#### Senate holds rare Saturday session aimed at ending shutdown\n\nSenators convened for a rare Saturday session aimed at ending the government shutdown, with no signs of an imminent breakthrough."}, {'url': 'https://www.searchenginejournal.com/google-algorithm-history/', 'content': 'Google Concludes Rollout Of September 2023 Helpful Content Update ⇾\n\nGoogle’s Mueller Outlines Path To Recovery For Sites Hit By Core Update ⇾\n\n## August 2023 Core Update\n\nGoogle announced a core algorithm update, which began on August 22 and concluded on September 7. To learn more about core updates, see the Google Search Central Blog.\n\nGoogle Launches August 2023 Core Update ⇾\n\n## April 2023 Reviews Update [...] Google Launches June 2024 Spam Update ⇾\n\nGoogle Completes June 2024 Spam Update Rollout ⇾\n\n## AI Overviews\n\nGoogle introduces AI-generated summaries (previously known as SGE) to U.S. search results, utilizing the new Gemini model designed specifically for search.\n\nGoogle Rolls Out AI-Powered Overviews To US Search Results ⇾\n\nGoogle’s AI Overviews Documentation: Key SEO Insights ⇾\n\nGoogle Search Expands AI Overviews Globally ⇾\n\n## Site Reputation Abuse [...] Google’s August Core Update Rollout Completed ⇾\n\nGoogle’s Guidance About The Recent Ranking Update ⇾\n\n## Explicit Fake Content Update\n\nUpdate to Google’s search algorithm to address non-consensual explicit content, specifically targeting artificially generated images and videos known as ‘deepfakes.’\n\nGoogle Search Algorithm Update Targeting Explicit Deepfakes ⇾\n\n## June 2024 Spam Update\n\nGoogle has started implementing the June 2024 spam update, which is expected to take a week to finish.'}, {'url': 'https://status.search.google.com/products/rGHU1u87FJnkP6W2GwMi/history', 'content': '# Google Search Status Dashboard\n\n## All incidents reported for Ranking\n\n## 2025\n\n| Summary | Date | Duration |\n --- \n| August 2025 spam update | 26 Aug 2025 | 26 days, 15 hours |\n| June 2025 core update | 30 Jun 2025 | 16 days, 18 hours |\n| March 2025 core update | 13 Mar 2025 | 13 days, 21 hours |\n\n## 2024'}]I have found several recent news articles about Google. Now I will analyze the sentiment of each headline to determine if the news is positive.
Action: SentimentAnalyzer
Action Input: "Update on Google's quantum computer"Tool: Loading sentiment analysis model for the first time...
Hardware accelerator e.g. GPU is available in the environment, but no `device` argument is passed to the `Pipeline` object. Model will be on CPU.
Tool: Analyzing sentiment for text: 'Update on Google's quantum computer...'
Sentiment: neutral, Score: 0.7748Action: SentimentAnalyzer
Action Input: "OpenAI challenges Google Chrome"Tool: Analyzing sentiment for text: 'OpenAI challenges Google Chrome...'
Sentiment: neutral, Score: 0.7458Action: SentimentAnalyzer
Action Input: "Google to spend $1 billion on AI education and job training in U.S."Tool: Analyzing sentiment for text: 'Google to spend $1 billion on AI education and job...'
Sentiment: positive, Score: 0.6160Action: SentimentAnalyzer
Action Input: "Google DeepMind CEO demonstrates Genie 2"Tool: Analyzing sentiment for text: 'Google DeepMind CEO demonstrates Genie 2...'
Sentiment: neutral, Score: 0.7280Action: SentimentAnalyzer
Action Input: "Google, OpenAI, Spotify and other platforms hit by widespread outage"Tool: Analyzing sentiment for text: 'Google, OpenAI, Spotify and other platforms hit by...'
Sentiment: positive, Score: 0.5665An error occurred: An output parsing error occurred. In order to pass this error back to the agent and have it try again, pass `handle_parsing_errors=True` to the AgentExecutor. This is the error: Could not parse LLM output: `I have found the latest news about Google and analyzed the sentiment of the headlines. Here is a summary:

*   **Positive News:**
    *   Google announced it will spend $1 billion on AI education and job training in the U.S., partnering with colleges to give students free access to its AI tools.

*   **Neutral News:**
    *   There is an update on Google's quantum computer, which is reported to be significantly faster than traditional supercomputers.
    *   OpenAI is presenting a challenge to Google Chrome.
    *   The CEO of Google DeepMind demonstrated a new AI model called Genie 2.

*   **Negative News:**
    *   Google, along with other major platforms like OpenAI and Spotify, was affected by a widespread tech outage.

In summary, the latest news about Google is a mix of positive, neutral, and negative events. The most significant positive news is their major investment in AI education, while the most notable negative event was the recent service outage.`
```

## AI Governance Considerations:
* Hallucination Mitigation: The agent is explicitly prompted to use local data first and to cite its sources. The FinancialDocumentAnalyst tool is instructed to answer only from the provided file.
* Data Privacy: The current implementation processes local files. For a production system, care must be taken to ensure no proprietary data is sent to third-party APIs without proper anonymization or contractual safeguards.

## Strategic Roadmap & future TODOs:
* Governance & Validation: Introduce a human-in-the-loop validation step where the agent's synthesized reports are flagged for review before being finalized, ensuring accuracy and compliance.