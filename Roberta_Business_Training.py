from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
import torch
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


df = pd.read_csv('/content/Annual_Reports_With_Text.csv')


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df['Stock Price Change'] = scaler.fit_transform(df[['Stock Price Change']])
df['Stock Price Change'] = df['Stock Price Change'].astype(float)
dataset = Dataset.from_pandas(df)

model_name = "/content/Business_Model"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForSequenceClassification.from_pretrained(model_name,num_labels = 1, problem_type="regression", ignore_mismatched_sizes=True)
model = model.float()
def tokenize_function(examples):
    return tokenizer(examples["Report_Text"], padding="max_length", truncation=True)


tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Set the format to PyTorch tensors and specify the data type for labels
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask"], output_all_columns=True)
tokenized_dataset = tokenized_dataset.rename_column("Stock Price Change", "label")
print(tokenized_dataset.column_names)
tokenized_dataset = tokenized_dataset.map(lambda x: {"label": torch.tensor(float(x["label"]), dtype=torch.float32)})
tokenized_dataset = tokenized_dataset.remove_columns(["Report_Text"])
#tokenized_dataset = tokenized_dataset.rename_column("Stock Price Change", "label")
tokenized_dataset.set_format("torch")


split_dataset = tokenized_dataset.train_test_split(test_size=0.2)


from transformers import EvalPrediction
import numpy as np

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    mse = np.mean((predictions.squeeze() - labels) ** 2)
    rmse = np.sqrt(mse)
    return {"mse": mse, "rmse": rmse}

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=6,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    evaluation_strategy="epoch",
    save_strategy="epoch",  
    load_best_model_at_end=True,
    metric_for_best_model="mse",
    fp16=True,
    learning_rate=5e-5
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=split_dataset["train"],
    eval_dataset=split_dataset["test"],
    compute_metrics=compute_metrics,
)
trainer.train()


model.save_pretrained("fine_tuned_roberta_stock_predictor")
tokenizer.save_pretrained("fine_tuned_roberta_stock_predictor")