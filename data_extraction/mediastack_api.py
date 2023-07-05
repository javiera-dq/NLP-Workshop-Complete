# Python 3
import http.client, urllib.parse
from dotenv import load_dotenv
import os
import json

# Get the current directory of your_file.py
current_dir = os.path.dirname(os.path.abspath(__file__))
# Define the path to the data folder
data_folder = os.path.join(current_dir, '..', 'data')

load_dotenv()
# Set up your Twitter API credentials
ACCESS_KEY = os.getenv('mediastack_access_key')
conn = http.client.HTTPConnection('api.mediastack.com')

params = urllib.parse.urlencode({
    'access_key': ACCESS_KEY,
    'categories': 'general,technology',
    'languages': 'en',
    'keywords': 'emobility lithium-ion',
    'sort': 'published_desc',
    # 'countries': 'de',
    'limit': 100, # fetching 100 news files
    })

conn.request('GET', '/v1/news?{}'.format(params))

res = conn.getresponse()
data = res.read().decode('utf-8')  # Decode the byte object into a string

# Extract the relevant fields
result = []
for item in json.loads(data)['data']:  # Parse the JSON string
    entry = {
        "title": item['title'],
        "text": item['description'],
        "url": item['url'],
        "source": item['source'],
        "category": item['category'],
        "published_at": item['published_at']
    }
    result.append(entry)

# Convert the result to JSON
json_result = json.dumps(result, indent=4)

json_file_path_export = os.path.join(data_folder, 'news_api_data.json')
# Save the scraped data to a new JSON file
with open(json_file_path_export, 'w') as file:
    json.dump(json_result, file, indent=4)
