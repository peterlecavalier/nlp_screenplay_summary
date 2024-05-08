'''
Remove metadata (any line with "M:" tag) from scripts
'''

from glob import glob
import os
from tqdm import tqdm

MOVIES_FP = "."

fps = glob(f"{MOVIES_FP}/scripts/parsed/tagged/*.txt")

for fp in tqdm(fps):
    with open(fp, encoding="utf-8") as tagged_file:
        new_file = open(f"{MOVIES_FP}/scripts/parsed/final/{fp.split(os.sep)[-1][:-10]}final.txt", "a", encoding='utf-8')
        for line in tagged_file:
            if line[:2] != "M:":
                new_file.write(line)
                
        new_file.close()