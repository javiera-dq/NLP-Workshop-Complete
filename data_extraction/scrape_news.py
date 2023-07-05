from bs4 import BeautifulSoup
import requests
import json
import os

# Get the current directory of your_file.py
current_dir = os.path.dirname(os.path.abspath(__file__))
# Define the path to the data folder
data_folder = os.path.join(current_dir, '..', 'data')
# Define the JSON file path
json_file_path_import = os.path.join(data_folder, 'news_api_data.json')

def scrape_webpage(url):
    # Send a GET request to the URL
    response = requests.get(url)

    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')
    
    content = soup.get_text()
    return(content)


# Open and read the JSON file
with open(json_file_path_import, 'r') as json_file:
    data = json.load(json_file)


# Iterate through each item, scrape the text content, and save it in a new JSON file
scraped_data = []
for item in data:
    url = item['url']
    text_content = scrape_webpage(url)
    item['text_content'] = text_content
    scraped_data.append(item)

json_file_path_export = os.path.join(data_folder, 'scraped_data.json')
# Save the scraped data to a new JSON file
with open(json_file_path_export, 'w') as file:
    json.dump(scraped_data, file, indent=4)

print("Scraped data saved to scraped_data.json")