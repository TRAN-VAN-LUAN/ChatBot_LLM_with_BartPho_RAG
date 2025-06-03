import os
import json
import requests
from bs4 import BeautifulSoup
import csv

# Load dữ liệu JSON từ file
def load_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None

def scrape_page_complete(url, json_title):
    """
    Lấy toàn bộ nội dung từ trang web dựa trên `url` theo thứ tự từ trên xuống dưới
    và trả về một cặp (tiêu đề, nội dung), không thêm tiền tố và tránh nội dung bị lặp.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Sử dụng tiêu đề từ JSON hoặc từ thẻ h1 của trang
        title = json_title
        h1_tag = soup.find('h1')
        if h1_tag:
            title = h1_tag.get_text(strip=True)

        # Thu thập nội dung chính của trang
        main_content = soup.find('div', attrs={'data-testid': 'topic-main-content'})
        
        if not main_content:
            # Thử tìm theo cách khác nếu không có div chính
            content_sections = soup.find_all('section', attrs={'data-testid': lambda x: x and x != 'footer'})
            
            if content_sections:
                main_content = BeautifulSoup("<div></div>", "html.parser").div
                for section in content_sections:
                    skip_section = False
                    for span in section.find_all('span'):
                        if "Tài liệu tham khảo" in span.get_text():
                            skip_section = True
                            break
                    
                    if not skip_section:
                        main_content.append(section)
        
        if not main_content:
            print(f"No main content found for {url}")
            return None
            
        # Loại bỏ các phần không cần thiết
        for element in main_content.select("[data-testid='topicFooterWrapper'], footer, .footer, #footer"):
            element.decompose()
        
        for element in main_content.select("script, style, meta, link, noscript"):
            element.decompose()
        
        # Xóa phần tham khảo
        for span in main_content.find_all('span'):
            if "Tài liệu tham khảo" in span.get_text():
                parent = span.find_parent('section') or span.find_parent('div')
                if parent:
                    parent.decompose()
                else:
                    span.decompose()
        
        # === Phương pháp mới: Lấy nội dung tuần tự một cách đơn giản ===
        # Thu thập tất cả văn bản theo thứ tự xuất hiện trong DOM
        all_text = []
        processed_text = set()  # Để tránh lặp lại
        
        # Làm phẳng cây DOM và lấy văn bản theo thứ tự xuất hiện
        def get_text_nodes(element):
            result = []
            # Nếu phần tử có văn bản trực tiếp và không chỉ là khoảng trắng
            if element.string and element.string.strip():
                result.append((element, element.string.strip()))
            
            # Duyệt qua tất cả con trực tiếp
            for child in element.children:
                if isinstance(child, str) and child.strip():
                    result.append((element, child.strip()))
                elif hasattr(child, 'children'):
                    result.extend(get_text_nodes(child))
            
            return result
        
        # Lấy tất cả các node văn bản và phần tử chứa chúng
        text_nodes = get_text_nodes(main_content)
        
        # Sắp xếp node theo vị trí xuất hiện trong trang
        for element, text in text_nodes:
            # Chỉ thêm văn bản có ý nghĩa và chưa xuất hiện trước đó
            if text and len(text) > 1 and text not in processed_text:
                all_text.append(text)
                processed_text.add(text)
        
        # Kết hợp tất cả thành một nội dung hoàn chỉnh
        content = " ".join(all_text)  # Sử dụng khoảng trắng thay vì xuống dòng
        
        # Làm sạch nội dung
        content = content.replace('\xa0', ' ')
        content = content.replace('\u200b', '')
        content = content.replace('\n', ' ')  # Thay thế bất kỳ xuống dòng nào còn lại
        
        # Thêm dấu cách sau dấu câu
        for punct in '.,:;!?':
            content = content.replace(f'{punct}', f'{punct} ')
            content = content.replace(f'{punct}  ', f'{punct} ')
        
        # Loại bỏ dấu cách thừa
        while "  " in content:
            content = content.replace("  ", " ")
        
        # Chỉ trả về nếu có đủ nội dung
        if len(content) >= 100:
            return title, content.strip()
        
        return None
    
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None
        print(f"Error fetching {url}: {e}")
        return None

# Tạo file CSV và ghi dữ liệu
def save_data_to_csv(data, file_path='E:/university/TLCN/ChatBot/data/csv/medical_complete.csv'):
    with open(file_path, mode='w', newline='', encoding='utf-8') as file:
        # Sử dụng dấu tab làm delimiter thay vì dấu phẩy để tránh xung đột
        # Đồng thời đảm bảo quoting toàn bộ nội dung để bảo toàn xuống dòng
        csv_writer = csv.writer(file, delimiter='\t', quotechar='"', quoting=csv.QUOTE_ALL)
        
        # Ghi tiêu đề cột
        csv_writer.writerow(['Title', 'Complete_Content', 'Source_URL'])
        
        # Gọi hàm để xử lý dữ liệu
        process_data(data, csv_writer)
        
        print(f"\nData successfully saved to {file_path}")

def process_data(data, csv_writer):
    for item in data:
        title = item.get("title", item.get("subTitle", ""))
        link = item.get("link")
        
        print(f"Processing: {title} - {link}")

        if "children" not in item or len(item["children"]) == 0:
            # Cào toàn bộ trang thay vì từng phần
            result = scrape_page_complete(link, title)
            if result:
                final_title, complete_content = result
                
                # Xử lý nội dung để tương thích với CSV
                # 1. Loại bỏ các ký tự đặc biệt
                complete_content = complete_content.replace('\xa0', ' ')
                complete_content = complete_content.replace('\u200b', '')
                
                # 2. Đảm bảo dấu nháy kép được escape đúng cách (tránh vấn đề với quoting)
                complete_content = complete_content.replace('"', '""')
                
                # 3. Thay thế bất kỳ ký tự xuống dòng nào còn lại bằng khoảng trắng
                complete_content = complete_content.replace('\n', ' ')
                
                # 4. Loại bỏ khoảng trắng dư thừa
                while "  " in complete_content:
                    complete_content = complete_content.replace("  ", " ")
                
                # Ghi vào CSV
                csv_writer.writerow([final_title, complete_content, link])
                print(f"✅ Added entry: {final_title} - Length: {len(complete_content)} characters")
            else:
                print(f"❌ No useful content found for: {link}")
        else:
            # Đệ quy xử lý các mục con
            process_data(item["children"], csv_writer)
        

# Load JSON data from a file
json_file_path = 'E:/university/TLCN/ChatBot/data/json/listData.json'
json_data = load_json_file(json_file_path)

if json_data:
    # Save scraped data to CSV
    save_data_to_csv(json_data)
else:
    print("Failed to load or process JSON data.")