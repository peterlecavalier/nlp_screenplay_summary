import requests
import re

S = requests.Session()

URL = "https://en.wikipedia.org/w/api.php"

PARAMS = {
    "action": "opensearch",
    "namespace": "0",
    "search": "WALL·E",
    "limit": "1",
    "format": "json"
}

R = S.get(url=URL, params=PARAMS)
query_data = R.json()
print(query_data)