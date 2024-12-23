from datasets import Dataset
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=3)  # Adjust num_labels based on your data

# Load CSV file into pandas DataFrame
business_df = pd.read_csv("/content/all-data.csv")
label_mapping = {"positive": 0, "negative": 1, "neutral": 2}
business_df["Sentiment"] = business_df["Sentiment"].map(label_mapping)

# Convert DataFrame to Hugging Face Dataset
dataset = Dataset.from_pandas(business_df)

# Split dataset into train/validation/test sets
train_testvalid = dataset.train_test_split(test_size=0.3)
test_valid = train_testvalid['test'].train_test_split(test_size=0.5)

train_dataset = train_testvalid['train']
valid_dataset = test_valid['train']
test_dataset = test_valid['test']

# Define tokenization function
def tokenize_function(examples):
    tokenized = tokenizer(examples['Text'], padding="max_length", truncation=True)
    tokenized['labels'] = examples['Sentiment']
    return tokenized
# Tokenize datasets
tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=["Text", "Sentiment"])
tokenized_valid = valid_dataset.map(tokenize_function, batched=True, remove_columns=["Text", "Sentiment"])
tokenized_test = test_dataset.map(tokenize_function, batched=True, remove_columns=["Text", "Sentiment"])

print(tokenized_train.column_names)
print(tokenized_valid.column_names)
print(tokenized_test.column_names)
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=10,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=32,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    evaluation_strategy="epoch",  # Evaluate at each epoch
    save_strategy="epoch",        # Save model at each epoch
    load_best_model_at_end=True,  # Load best model based on validation loss
    remove_unused_columns=False   # Keep all necessary columns
)
from transformers import Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_valid,
    tokenizer=tokenizer
)
trainer.train()
model.save_pretrained("fine_tuned_roberta_business")
tokenizer.save_pretrained("fine_tuned_roberta_business")