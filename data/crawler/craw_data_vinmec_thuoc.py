import json
import csv
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# Load the JSON file
with open('e:/university/TLCN/ChatBot/data/json/vinmec_medicine.json', 'r', encoding='utf-8') as file:
    links = json.load(file)

# Base URL for the links
base_url = "https://vinmec.com"

def extract_columns(soup, columns):
    data = {column: '' for column in columns}
    # Find all <h2> tags with class 'title_detail_sick'
    h2_tags = soup.find_all('h2', class_='title_detail_sick')
    for h2 in h2_tags:
        text = h2.get_text().strip()
        for column in columns:
            if text.lower().startswith(column.lower()):
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
csv_file = 'e:/university/TLCN/ChatBot/data/csv/vinmec_thuoc_data.csv'
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

columns = ['Dạng bào chế - biệt dược', 'Nhóm thuốc – Tác dụng', 'Chỉ định', 'Chống chỉ định', 'Thận trọng', 'Tác dụng không mong muốn',
       'Liều và cách dùng', 'Chú ý khi sử dụng']
print('columns:', columns)

with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=['link', 'Title'] + columns)
    writer.writeheader()

    for link in links:
        full_link = base_url + link
        print(f"Processing {full_link}")
        response = requests.get(full_link)
        soup = BeautifulSoup(response.content, 'html.parser')

        # Hàm trích xuất tiêu đề từ trang
        title = soup.title.string.strip() if soup.title else 'No Title'

        # Hàm tự định nghĩa để trích xuất dữ liệu từ các cột
        row = {'link': full_link, 'Title': title}
        row.update(extract_columns(soup, columns))
        writer.writerow(row)