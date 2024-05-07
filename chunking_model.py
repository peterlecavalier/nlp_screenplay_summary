import torch
import json
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from transformers import DataCollatorForSeq2Seq
from tqdm import tqdm

MOVIES_FP = "."

# Function to split dialogue into chunks
def split_into_chunks(text, chunk_size=1000):
    words = text.split()
    chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

# Custom dataset
class MovieDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        # Return input IDs and labels as dictionaries
        return {'input_ids': self.encodings[idx], 'labels': self.labels[idx]}

    def __len__(self):
        return len(self.labels)

# Load the tokenizer and model
model_name = 't5-small'
tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# Load metadata
with open(f'{MOVIES_FP}/scripts/metadata/clean_parsed_meta.json', 'r') as file:
    metadata = json.load(file)

# Prepare data for processing
input_ids = []
labels = []

# Parse our metadata to create new dict of dialog and plot summary (from Wikipedia)
for key in tqdm(metadata.keys(), desc="Processing metadata"):
    if ('Summary' not in metadata[key]) or (metadata[key]['Summary'] is None):
        continue

    file_name = metadata[key]['file']['file_name']
    with open(f"{MOVIES_FP}/scripts/parsed/final/{file_name}_final.txt", 'r') as tagged_file:
        tagged = tagged_file.read().lower()

    tagged_chunks = split_into_chunks(tagged)
    chunk_summaries = []
    # Process chunks in batches
    for i in range(0, len(tagged_chunks), 10):
        batch = tagged_chunks[i:i+10]
        batch_inputs = tokenizer(batch, max_length=512, truncation=True, padding='longest', return_tensors="pt").to(model.device)
        summaries = model.generate(**batch_inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
        chunk_summaries.extend(tokenizer.batch_decode(summaries, skip_special_tokens=True))

    combined_summary = ' '.join(chunk_summaries)
    tokenized_combined_summary = tokenizer.encode(combined_summary, truncation=True, padding="max_length", max_length=128)
    tokenized_label = tokenizer.encode(metadata[key]['Summary'].lower(), truncation=True, padding="max_length", max_length=128)

    input_ids.append(tokenized_combined_summary)
    labels.append(tokenized_label)


# Assuming tokenized_input and tokenized_label are lists containing the inputs and labels
train_dataset = MovieDataset(input_ids, labels)