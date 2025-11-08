import numpy as np
import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, f1_score

# --- CONFIGURATION ---
MODEL_NAME = "distilbert/distilbert-base-uncased"
DATASET_NAME = "financial_phrasebank"
# Adjusted the output directory to be relative to the project root
OUTPUT_DIR = "./models/sentiment_analyzer"

def main():
    """
    This script fine-tunes a sentiment analysis model on financial data
    and saves it to a local directory.
    """
    print("--- Starting Sentiment Model Fine-Tuning ---")

    # 1. Load Dataset
    print(f"Loading dataset: {DATASET_NAME}...")
    # We use the 'sentences_allagree' split for higher quality data
    dataset = load_dataset(DATASET_NAME, "sentences_allagree", split="train")

    # To make training faster for this demonstration, let's select a subset of the data.
    dataset = dataset.select(range(1000)) # Using 1000 examples for a quick run.

    # The labels are 0: negative, 1: neutral, 2: positive. We need to map them.
    labels = ["negative", "neutral", "positive"]
    id2label = {i: label for i, label in enumerate(labels)}
    label2id = {label: i for i, label in enumerate(labels)}

    # 2. Load Tokenizer and Model
    print(f"Loading model and tokenizer: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=3, id2label=id2label, label2id=label2id
    )

    # 3. Preprocess Data
    print("Preprocessing dataset...")
    def preprocess_function(examples):
        return tokenizer(examples["sentence"], truncation=True, padding="max_length")

    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    # 4. Set up Trainer
    print("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=2e-5,
        num_train_epochs=1, # One epoch is enough to demonstrate the process
        weight_decay=0.01,
        evaluation_strategy="no", # We don't need evaluation for this project
        save_strategy="epoch",
        no_cuda=True,  # This is a more forceful flag to disable all GPU/MPS usage
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_dataset)

    # 5. Train and Save
    print("--- Starting Training ---")
    trainer.train()
    print("--- Training Complete ---")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model and tokenizer saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
