from transformers import pipeline
from pathlib import Path

# Path to the fine-tuned model we just trained
MODEL_PATH = Path.cwd() / "models" / "sentiment_analyzer"

# We will load the pipeline lazily (only when first used) to prevent startup errors.
sentiment_pipeline = None

def get_sentiment(text: str) -> str:
    """
    Analyzes the sentiment of a given text using a fine-tuned financial sentiment model.
    Input should be a single news headline or a short text snippet.
    Returns the sentiment (positive, negative, or neutral) and the confidence score.
    """
    global sentiment_pipeline
    if not MODEL_PATH.exists():
        return "Tool Error: The fine-tuned sentiment model was not found. Please run train_sentiment_model.py to enable this tool."

    if sentiment_pipeline is None:
        print("Tool: Loading sentiment analysis model for the first time...")
        sentiment_pipeline = pipeline("sentiment-analysis", model=str(MODEL_PATH), tokenizer=str(MODEL_PATH))

    print(f"Tool: Analyzing sentiment for text: '{text[:50]}...'")
    result = sentiment_pipeline(text)
    # The pipeline returns a list, so we access the first element.
    return f"Sentiment: {result[0]['label']}, Score: {result[0]['score']:.4f}"