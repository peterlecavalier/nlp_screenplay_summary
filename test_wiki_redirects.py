import requests
import re

S = requests.Session()

URL = "https://en.wikipedia.org/w/api.php"

PARAMS = {
        "action": "query",
        "format": "json",
        "titles": "The_Imitation_Game_(2014_film)",
        "prop": "info",
    }

R = S.get(url=URL, params=PARAMS)
query_data = R.json()
pages = query_data["query"]["pages"]
page_id = next(iter(pages))
page_info = pages[page_id]
print(page_info)