import json
from glob import glob

save_file = open("/mnt/disks/main_storage/Movie-Script-Database/not_present_movies.txt", "a")

txt_files = glob("/mnt/disks/main_storage/Movie-Script-Database/scripts/parsed/tagged/*.txt")

with open('/mnt/disks/main_storage/Movie-Script-Database/scripts/metadata/clean_parsed_meta.json') as json_file:
    data = json.load(json_file)

for key in data.keys():
    if 'parsed' not in data[key].keys():
        save_file.write(f"{data[key]['parsed']['tagged']}\n")
    elif f"/mnt/disks/main_storage/Movie-Script-Database/scripts/parsed/tagged/{data[key]['parsed']['tagged']}" not in txt_files:
        save_file.write(f"{data[key]['parsed']['tagged']}\n")