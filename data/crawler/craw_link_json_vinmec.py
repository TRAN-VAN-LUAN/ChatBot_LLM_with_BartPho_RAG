import json
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# Load the JSON file
with open('e:/university/TLCN/ChatBot/data/json/vinmec.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Base URL for the links
base_url = "https://vinmec.com"

def extract_links(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find the ul with class 'list_result_AZ flex'
    ul = soup.find('ul', class_='list_result_AZ flex')
    links = []
    if ul:
        # Extract all <a> tags within the ul
        a_tags = ul.find_all('a')
        for a in a_tags:
            href = a.get('href')
            if href and not href.startswith('javascript'):
                full_url = urljoin(base_url, href)
                title = a.get_text().split(':')[0].strip()
                links.append({'link': full_url, 'title': title})
    
    return links, soup

# Iterate through each link in the JSON file
for item in data['links']:
    url = urljoin(base_url, item['link'])
    links, soup = extract_links(url)
    item['children'].extend(links)
    

# Save the updated JSON file
with open('e:/university/TLCN/ChatBot/data/json/vinmec.json', 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=4)