import pandas as pd
from datasets import Dataset
import datasets
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer , GenerationConfig, TrainingArguments, Trainer
import time
import torch
from CleanText import clean_text
from TokenizeFunction import tokenize_function
from ModelTokenizer import get_tokenizer , get_model #check

import json
# Detect the GPU if any; if not, use CPU. If on a Mac with M1/M2, consider using MPS.
device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open('config.json', 'r') as file:
  config = json.load(file)

# Load the dataset into a Pandas DataFrame, selecting only the 'question' and 'answer' columns
data = pd.read_csv("NvidiaDocumentationQandApairs.csv")[["question", "answer"]]
model_name = config['model_name']
# Print the shape of the dataset and display the first few rows
# print(data.shape)
# print(data.head())

data['question'] = data['question'].apply(clean_text)
data['answer'] = data['answer'].apply(clean_text)

train=data.sample(frac=0.7,random_state=7) # Create training of 70% of the data
test=data.drop(train.index) # Create testing by removing the 70% of the train data which will result in 30%

val=test.sample(frac=0.5,random_state=7) # Create validation of 50% of the testing data
test=test.drop(val.index) # Create testing by removing the 50% of the validation data which will result in 50%


model = get_model(model_name)
tokenizer = get_tokenizer(model_name)

train_data = Dataset.from_pandas(train)
train_tokenized_datasets = train_data.map(tokenize_function, batched=True)
train_tokenized_datasets = train_tokenized_datasets.remove_columns(['question', 'answer','__index_level_0__'])


val_data = Dataset.from_pandas(val)
val_tokenized_datasets = val_data.map(tokenize_function, batched=True)
val_tokenized_datasets = val_tokenized_datasets.remove_columns(['question', 'answer','__index_level_0__'])


test_data = Dataset.from_pandas(test)
test_tokenized_datasets = test_data.map(tokenize_function, batched=True)
test_tokenized_datasets = test_tokenized_datasets.remove_columns(['question', 'answer','__index_level_0__'])

EPOCHS = config["EPOCHS"] #1
LR = config["LR"] #1e-4
BATCH_SIZE = config["BATCH_SIZE"] #2

training_path = config["training_path"]#"./training_path_nvidia_chatbot"


training_args = TrainingArguments(
    output_dir = training_path,
    overwrite_output_dir = True,
    per_device_train_batch_size = BATCH_SIZE,
    per_device_eval_batch_size = BATCH_SIZE,
    learning_rate = LR,
    num_train_epochs = EPOCHS,
    evaluation_strategy = "epoch",
    save_total_limit = 2
    )

trainer= Trainer(
    model = model,
    args = training_args,
    train_dataset = train_tokenized_datasets,
    eval_dataset = val_tokenized_datasets,
)


trainer.train()

model_path = config["model_path"]#"./nvidia-chatbot-final-model"

trainer.model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
