import requests
import re

S = requests.Session()

URL = "https://en.wikipedia.org/w/api.php"

PARAMS = {
    "action": "parse",
    "format": "json",
    "page": "Rebel_Without_a_Cause",
    "prop": "wikitext",
    "section": 1,
    "disabletoc": 1}

### First check - "MOVIE_NAME RELEASE_YEAR film"
R = S.get(url=URL, params=PARAMS)
query_data = R.json()
summary = query_data['parse']['wikitext']['*']

# Replace file references
def replace_files(replace_str, start="[[", end="]]", start_tag="File:"):
    while replace_str.find(start + start_tag) != -1:
        start_idx = replace_str.find(start + start_tag)
        # Make a stack to replace
        stack = [start_idx]
        cur_idx = start_idx + len(start) + len(start_tag)
        while len(stack) != 0:
            next_start = replace_str.find(start, cur_idx)
            next_end = replace_str.find(end, cur_idx)
            if next_start != -1 and next_start < next_end:
                stack.append(next_start)
                cur_idx = next_start + len(start)
            elif next_end != -1:
                stack.pop()
                cur_idx = next_end + len(end)
            else:
                raise Exception("Something wrong happened in file removal!")
        
        replace_str = replace_str[:start_idx] + replace_str[cur_idx:]
    return replace_str

summary = replace_files(summary)
summary = replace_files(summary, start_tag="Image:")
summary = replace_files(summary, r"{{", r"}}", "quote")
summary = replace_files(summary, r"{{", r"}}", "#tag")
summary = replace_files(summary, r"{{", r"}}", "Hatnote")

# Replace any unit conversions
def replace_conversions(replace_str):
    while replace_str.find(r"{{convert|") != -1 or replace_str.find(r"{{cvt|") != -1:
        if replace_str.find(r"{{convert|") != -1:
            start_idx = replace_str.find(r"{{convert|")
            text_start_idx = start_idx + len(r"{{convert|")
        else:
            start_idx = replace_str.find(r"{{cvt|")
            text_start_idx = start_idx + len(r"{{cvt|")
        end_idx = replace_str.find(r"}}", text_start_idx)

        first_sep_idx = replace_str.find('|', text_start_idx)


        second_sep_idx = replace_str.find('|', first_sep_idx + 1)
        if second_sep_idx == -1 or second_sep_idx > end_idx:
            second_sep_idx = end_idx

        replace_str = replace_str[:start_idx] + replace_str[text_start_idx:first_sep_idx] + \
            " " + replace_str[first_sep_idx + 1:second_sep_idx] + replace_str[end_idx + 2:]
    return replace_str 

summary = replace_conversions(summary)

# Replace enclosed statements with the specified start and end
def replace_enclosed(replace_str, start, end, alt=None):
    while replace_str.find(start) != -1:
        start_idx = replace_str.find(start)
        end_idx = replace_str.find(end, start_idx + len(start))
        if alt is not None:
            alt_idx = replace_str.find(alt, start_idx + len(start))
            if alt_idx != -1 and (end_idx == -1 or alt_idx < end_idx):
                end_idx = alt_idx
                cur_end_len = len(alt)
            else:
                cur_end_len = len(end)
        else:
            cur_end_len = len(end)
        
        replace_str = replace_str[:start_idx] + replace_str[end_idx + cur_end_len:]
    
    return replace_str

summary = replace_enclosed(summary, r"{{", r"}}")
summary = replace_enclosed(summary, "<!--", "-->")
summary = replace_enclosed(summary, "<ref", "</ref>", alt="/>")
summary = replace_enclosed(summary, "===", "===")
summary = replace_enclosed(summary, "==", "==")

# Replace substitute phrases, denoted in wikitext as [[original link|new text]]
# Also removes any links at all, denoted in wikitext as [[link to other page]]
def replace_substitutes(replace_str, start="[[", sep="|", end="]]"):
    while replace_str.find(start) != -1:
        start_idx = replace_str.find(start)
        sep_idx = replace_str.find(sep)
        end_idx = replace_str.find(end)

        if sep_idx != -1 and sep_idx < end_idx:
            # Replace substitutes with the raw page text
            replace_str = replace_str[:start_idx] + replace_str[sep_idx + len(sep):end_idx] + replace_str[end_idx + len(end):]
        else:
            replace_str = replace_str[:start_idx] + replace_str[start_idx + len(start):end_idx] + replace_str[end_idx + len(end):]
    return replace_str 

summary = replace_substitutes(summary)

# Remove extra whitespace

summary = " ".join(summary.split())

print(summary)