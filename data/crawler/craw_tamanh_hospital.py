import json
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time

# Đọc dữ liệu từ file JSON
with open('../json/updated_disease_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Hàm để loại bỏ số thứ tự ở đầu chuỗi
def remove_numbering(text):
    return re.sub(r'^\d+(\.\d+)*\s*', '', text).strip()

def get_questions_and_answers(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        content_div = soup.find('section', class_='box_detail div_over')
        if not content_div:
            print(f"No content found for {url}")
            return []

        qa_list = []
        current_h2 = None
        current_h3 = None
        current_question = None
        current_answer = []

        for tag in content_div.find_all(['h2', 'h3', 'h4', 'p', 'li']):
            if tag.name == 'h2':
                current_h2 = remove_numbering(tag.get_text(strip=True))
                current_question = current_h2

            elif tag.name == 'h3':
                if current_h2 and not current_answer:
                    current_question = f"{current_h2} {remove_numbering(tag.get_text(strip=True))}"
                else:
                    current_question = remove_numbering(tag.get_text(strip=True))

            elif tag.name == 'h4':
                if current_h3 and not current_answer:
                    current_question = f"{current_h3} {remove_numbering(tag.get_text(strip=True))}"
                else:
                    current_question = remove_numbering(tag.get_text(strip=True))
                current_h3 = tag.get_text(strip=True)

            elif tag.name in ['p', 'li'] and current_question:
                current_answer.append(remove_numbering(tag.get_text(strip=True)))

            if current_question and tag.name not in ['h2', 'h3', 'h4']:
                qa_list.append((current_question, ' '.join(current_answer)))
                current_answer = []
                current_question = None

        if current_question and current_answer:
            qa_list.append((current_question, ' '.join(current_answer)))

        return qa_list

    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return []

# Tạo danh sách chứa dữ liệu Q&A
all_qa_data = []

def process_item(item):
    if 'children' not in item or not item['children']:
        link = item.get('link')
        if link:
            print(f"Fetching data from: {link}")  # In ra link đang xử lý
            qa_data = get_questions_and_answers(link)
            all_qa_data.extend(qa_data)
            time.sleep(1)  # Thêm khoảng thời gian chờ giữa các yêu cầu
    else:
        for sub_item in item['children']:
            process_item(sub_item)

# Xử lý từng item trong dữ liệu
for item in data:
    process_item(item)

# Chuyển đổi dữ liệu thành DataFrame và lưu thành file CSV
df = pd.DataFrame(all_qa_data, columns=['Question', 'Answer'])
df.to_csv('../csv/tamanh_hosipital_dataset.csv', index=False, encoding='utf-8-sig')

print("Data extraction completed. Results saved to tamanh_hosipital_dataset.csv.")
