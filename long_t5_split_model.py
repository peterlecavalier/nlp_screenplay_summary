import torch
import json
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq, AutoTokenizer, LongT5Model
from tqdm import tqdm
import numpy as np
import os

MOVIES_FP = "../Movie-Script-Database"
DICT_FP = "./tokens.json"

# Function to split up tagged scripts and corresponding summaries
def split_into_chunks(tagged, summary, chunk_len):
    tagged_chunks = []
    summary_chunks = []

    # Calculate desired summary chunk length
    divisor = len(tagged) / chunk_len
    summary_chunk_len = int(np.floor(len(summary) / divisor))

    # Make new summary
    while len(summary) > 0:
        summary_chunks.append(summary[:summary_chunk_len])
        summary = summary[summary_chunk_len:]

    # Make new tagged
    while len(tagged) > 0:
        tagged_chunks.append(tagged[:chunk_len])
        tagged = tagged[chunk_len:]
    
    # return list of lists for tagged and then summary
    return tagged_chunks, summary_chunks

# Custom dataset
class MovieDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        # Return input IDs and labels as dictionaries
        return {'input_ids': self.encodings[idx], 'decoder_input_ids': self.labels[idx]}

    def __len__(self):
        return len(self.labels)
    
    def save_dataset(self, fp):
        save_dict = {"encodings": self.encodings, "labels": self.labels}
        with open(fp, 'w') as file:
            json.dump(save_dict, file, indent=4)

    def load_dataset(self, fp):
        with open(fp) as json_file:
            data = json.load(json_file)
        
        self.encodings = data["encodings"]
        self.labels = data["labels"]


# Load the tokenizer and model
model_name = "google/long-t5-local-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = LongT5Model.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# Load metadata
with open(f'{MOVIES_FP}/scripts/metadata/clean_parsed_meta.json', 'r') as file:
    metadata = json.load(file)

# Prepare data for processing
input_ids = []
labels = []

# Load the dict if present
if os.path.isfile(DICT_FP):
    train_dataset = MovieDataset(None, None)
    train_dataset.load_dataset(DICT_FP)
else:
    # Parse our metadata to create new dict of dialog and plot summary (from Wikipedia)
    for key in tqdm(list(metadata.keys()), desc="Processing metadata"):
        if ('Summary' not in metadata[key]) or (metadata[key]['Summary'] is None):
            continue

        file_name = metadata[key]['file']['file_name']
        with open(f"{MOVIES_FP}/scripts/parsed/final/{file_name}_final.txt", 'r', encoding='utf-8') as tagged_file:
            tagged = tagged_file.read().lower()

        summary = metadata[key]['Summary'].lower()

        if len(tagged) <= 1 or len(summary) <= 1:
            continue

        tagged_tokenized = tokenizer.encode(tagged, truncation=False, padding=False)
        summary_tokenized = tokenizer.encode(summary, truncation=False, padding=False)

        tagged_chunks, summary_chunks = split_into_chunks(tagged_tokenized, summary_tokenized, chunk_len=4096)

        for idx in range(len(tagged_chunks)):
            input_ids.append(tagged_chunks[idx])
            labels.append(summary_chunks[idx])

    # Assuming tokenized_input and tokenized_label are lists containing the inputs and labels
    train_dataset = MovieDataset(input_ids, labels)
    train_dataset.save_dataset(DICT_FP)


# Split into training and evaluation
# Splitting the dataset for training and evaluation (simple random split)
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, eval_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])


# Training!
training_args = TrainingArguments(
    output_dir="./t5_summarization",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    learning_rate=5e-5,
    gradient_checkpointing=True
)

# Data collator for padding
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Start fine-tuning
trainer.train()
