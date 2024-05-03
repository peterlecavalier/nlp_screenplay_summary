import json
from glob import glob
import requests
from tqdm import tqdm
import re

WIKI_API_URL = "https://en.wikipedia.org/w/api.php"

def find_film_wiki_link(query_str, S):
    """
    Finds the films wiki link given a specific query string
    and requests session
    """
    params = {
        "action": "opensearch",
        "namespace": "0",
        "search": query_str,
        "limit": "1",
        "format": "json",
    }
    R = S.get(url=WIKI_API_URL, params=params)
    query_data = R.json()
    if len(query_data[3]) > 0:
        return query_data[3][0]
    else:
        return None

def get_summary_wiki(wiki_link):
    """
    Finds the film summary given a wiki link
    """
    # Just want the page name
    page_str = wiki_link.split('/')[-1]

    ### First, get the Plot section index
    plot_index = None
    params = {
        "action": "parse",
        "format": "json",
        "page": page_str,
        "prop": "sections",
        "disabletoc": 1
    }
    R = S.get(url=WIKI_API_URL, params=params)
    query_data = R.json()

    # Find the index of the Plot section
    try:
        for toc_section in query_data['parse']['sections']:
            if toc_section['line'].lower() == "plot":
                plot_index = toc_section['index']
    except:
        return None
    
    # If the Plot section wasn't found, just return None
    if plot_index is None:
        return None
    else:
        ### Next, extract the data from this Plot section
        params = {
            "action": "parse",
            "format": "json",
            "page": page_str,
            "prop": "wikitext",
            "section": plot_index,
            "disabletoc": 1}
        R = S.get(url=WIKI_API_URL, params=params)
        query_data = R.json()

        # Get the summary
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

        return summary


S = requests.Session()

with open('test_dump.json') as json_file:
    data = json.load(json_file)

error_films = 0

for key in tqdm(list(data.keys())[691:]):
    if 'parsed' in data[key].keys():
        print(key)
        try:
            title = data[key]['tmdb']['title']
        except:
            print(key)
            error_films += 1
            continue

        year = data[key]['tmdb']['release_date'][:4]

        # Check for the link
        checks = [f"{title} {year} film", f"{title} {year}", f"{title} film", f"{title}"]

        link = None
        summary = None
        for check in checks:
            link = find_film_wiki_link(check, S)
            if link is not None:
                summary = get_summary_wiki(link)
                if summary is not None:
                    break

        # TODO: add if link is not None, else add to error counter
        data[key]['Summary'] = summary
        # TODO: If none of the above worked, just snag from IMDb

        with open('test_dump.json', 'w') as fp:
            json.dump(data, fp, indent=4)

print(f"Total error films: {error_films}")