import torch
import json
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from transformers import DataCollatorForSeq2Seq
from transformers import AutoTokenizer, LongT5Model

MOVIES_FP = "."

# Load the tokenizer and model
# model_name = 't5-small'
# tokenizer = T5Tokenizer.from_pretrained('t5-small')
# model = T5ForConditionalGeneration.from_pretrained('t5-small')

model_name = "google/long-t5-local-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = LongT5Model.from_pretrained(model_name)

# Load metadata
with open(MOVIES_FP + '/scripts/metadata/clean_parsed_meta.json', 'r') as file:
    metadata = json.load(file)

# Create input dict
input_ids = []
labels = []

# Parse our metadata to create new dict of dialog and plot summary (from Wikipedia)
for key in metadata.keys():
    # Skip entries that have no summary
    if 'Summary' not in metadata[key] or metadata[key]['Summary'] is None:
        continue

    file_name = metadata[key]['file']['file_name']
    with open(f"{MOVIES_FP}/scripts/parsed/final/{file_name}_final.txt", 'r') as tagged_file:
        tagged = tagged_file.read().lower()

    # Prepare for tokenizer input
    # TODO: Do we need "summarize"?
    tokenizer_input = f"summarize: {tagged}"
    tokenizer_label = metadata[key]['Summary'].lower()

    # Tokenize the dialog and summary
    tokenized_input = tokenizer.encode(tokenizer_input, truncation=True, padding="max_length", max_length=8192)
    tokenized_label = tokenizer.encode(tokenizer_label, truncation=True, padding="max_length", max_length=8192)

    # Append the tokenized data
    input_ids.append(tokenized_input)
    labels.append(tokenized_label)

class MovieDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        # Return input IDs and labels as dictionaries
        return {'input_ids': self.encodings[idx], 'labels': self.labels[idx]}

    def __len__(self):
        return len(self.labels)

# Assuming tokenized_input and tokenized_label are lists containing the inputs and labels
train_dataset = MovieDataset(input_ids, labels)

# Split into training and evaluation
# Splitting the dataset for training and evaluation (simple random split)
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, eval_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

training_args = TrainingArguments(
    output_dir="./long_t5_summarization",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    learning_rate=5e-5,
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