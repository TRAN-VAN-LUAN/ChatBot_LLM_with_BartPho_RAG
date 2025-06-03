import json
import csv
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# Load the JSON file
with open('e:/university/TLCN/ChatBot/data/json/vinmec.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Base URL for the links
base_url = "https://vinmec.com"

def extract_columns(soup, columns):
    data = {column: '' for column in columns}
    # Find all <h2> tags with class 'title_detail_sick'
    h2_tags = soup.find_all('h2', class_='title_detail_sick')
    for h2 in h2_tags:
        text = h2.get_text().strip()
        for column in columns:
            if column.lower() in text.lower():
                # Extract content from the <div> with class 'collapsible-target'
                collapsible_target = h2.find_next('div', class_='collapsible-target')
                if collapsible_target:
                    content = ''
                    for child in collapsible_target.children:
                        if child.name == 'p':
                            content += child.get_text().strip() + '\n'
                        elif child.name == 'ul':
                            for li in child.find_all('li'):
                                if li.p:
                                    content += li.p.get_text().strip() + '\n'
                                else:
                                    content += li.get_text().strip() + '\n'
                    data[column] = content.strip()
                break
    return data

# Prepare the CSV file
csv_file = 'e:/university/TLCN/ChatBot/data/csv/vinmec_data.csv'
# columns = set()

# # First pass to collect all possible columns
# for item in data['links']:
#     for child in item['children']:
#         response = requests.get(child['link'])
#         print(f"Processing {child['link']}")
#         child_soup = BeautifulSoup(response.content, 'html.parser')
#         ul_tags = child_soup.find_all('ul', class_='list_type_detail_sick')
#         for ul in ul_tags:
#             a_tags = ul.find_all('a')
#             for a in a_tags:
#                 text = a.get_text().strip().split(':')[0]
#                 columns.add(text)

columns = ['Đối tượng nguy cơ', 'Phòng ngừa', 'Triệu chứng', 'Tổng quan', 'Biện pháp điều trị', 'Đường lây truyền',
       'Nguyên nhân', 'Biện pháp chẩn đoán']
print('columns:', columns)

# Write the CSV file
with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=['link', 'Title'] + columns)
    writer.writeheader()

    # Second pass to write the data
    for item in data['links']:
        for child in item['children']:
            print(f"Processing {child['link']}")
            response = requests.get(child['link'])
            child_soup = BeautifulSoup(response.content, 'html.parser')
            row = {'link': child['link'], 'Title': child['title']}
            row.update(extract_columns(child_soup, columns))
            writer.writerow(row)