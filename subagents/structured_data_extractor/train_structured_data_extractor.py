import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    Trainer,
)

MODEL_NAME = "google/flan-t5-small"
DATASET_PATH = "./subagents/structured_data_extractor/dataset.jsonl"
OUTPUT_DIR = "./models/structured_data_extractor"

def main():
    """
    This script fine-tunes a structured data extractor model on a dataset
    and saves it to a local directory.
    """
    print("--- Starting structured data extractor model Fine-Tuning ---")

    # 1. Load Dataset
    print(f"Loading dataset from: {DATASET_PATH}...")
    dataset = load_dataset('json', data_files=DATASET_PATH, split='train')

    split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    # 2. Load Tokenizer and Model
    print(f"Loading model and tokenizer: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    # 3. Preprocess Data
    print("Preprocessing dataset...")
    
    def preprocess_function(examples):
        """Tokenizes the input text and target JSON for the Seq2Seq model."""
        # Tokenize the inputs
        model_inputs = tokenizer(examples["input"], max_length=1024, truncation=True)

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["output"], max_length=128, truncation=True)
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
    tokenized_eval_dataset = eval_dataset.map(preprocess_function, batched=True)

    # 4. Set up Trainer
    print("Setting up training arguments...")
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=5e-6,
        num_train_epochs=50,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        use_cpu=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        data_collator=data_collator,
    )

    # 5. Train and Save
    print("--- Starting Training ---")
    trainer.train()
    print("--- Training Complete ---")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model and tokenizer saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
