import json
from tqdm import tqdm
from transformers import LEDTokenizer, AutoTokenizer, LongformerTokenizer
import matplotlib.pyplot as plt
import numpy as np

#tokenizer = LEDTokenizer.from_pretrained("bakhitovd/led-base-7168-ml")
#tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
tokenizer = AutoTokenizer.from_pretrained("google/long-t5-local-base")

print(f"Max sequence length of this model: {tokenizer.model_max_length}")

with open('./scripts/metadata/clean_parsed_meta.json') as json_file:
    data = json.load(json_file)

tagged_lengths = []
summary_lengths = []

for key in tqdm(list(data.keys())):
    if 'Summary' not in data[key].keys() or data[key]['Summary'] is None:
        continue

    file_name = data[key]['file']['file_name']
    with open(f"./scripts/parsed/final/{file_name}_final.txt", 'r', encoding="utf-8") as tagged_file:
        tagged = tagged_file.read().lower()

    tagged_tokenized = tokenizer.encode(tagged, truncation=False, padding=False)
    summary_tokenized = tokenizer.encode(data[key]["Summary"].lower(), truncation=False, padding=False)

    tagged_lengths.append(len(tagged_tokenized))
    summary_lengths.append(len(summary_tokenized))

plt.figure()
plt.hist(tagged_lengths, range=(0, 125000))
plt.title("Token lengths of tagged screenplays")
plt.xlabel("Token Length")
plt.ylabel("Count")
plt.savefig('./screenplay_lengths.png')
plt.show()

plt.figure()
plt.hist(summary_lengths, range=(0, 2000))
plt.title("Token lengths of summaries")
plt.xlabel("Token Length")
plt.ylabel("Count")
plt.savefig('./summary_lengths.png')
plt.show()



print("Screenplays:")
print(f"Min = {np.min(tagged_lengths)}")
print(f"Max = {np.max(tagged_lengths)}")
print(f"Mean = {np.mean(tagged_lengths)}")
print(f"Standard Deviation = {np.std(tagged_lengths)}")

print("Summaries:")
print(f"Min = {np.min(summary_lengths)}")
print(f"Max = {np.max(summary_lengths)}")
print(f"Mean = {np.mean(summary_lengths)}")
print(f"Standard Deviation = {np.std(summary_lengths)}")