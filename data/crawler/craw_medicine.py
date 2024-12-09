import json
import requests
from bs4 import BeautifulSoup
import pandas as pd

# Hàm cào dữ liệu từ link
def scrape_details(link):
    response = requests.get(link)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Cào dữ liệu cho từng cột
    data = {
        "Title": soup.title.string.strip()  # Lấy title của trang
    }

    # Tìm tất cả các thẻ h2 có class là 'title_detail_sick'
    h2_elements = soup.find_all('h2', class_='title_detail_sick')
    
    for h2 in h2_elements:
        header_text = h2.get_text(strip=True)
        next_div = h2.find_next('div', class_='body collapsible-target')
        
        # Nếu thẻ <div> tiếp theo tồn tại, lấy tất cả các thẻ <p> bên trong
        if next_div:
            p_elements = next_div.find_all('p')
            content = "\n".join([p.get_text(strip=True) for p in p_elements])
            
            # Tùy thuộc vào nội dung của thẻ h2, lưu vào cột tương ứng
            if header_text == "Dạng bào chế - biệt dược":
                data['Dạng bào chế - biệt dược'] = content
            elif header_text == "Nhóm thuốc – Tác dụng":
                data['Nhóm thuốc – Tác dụng'] = content
            else:
                # Tạo cột mới nếu tiêu đề h2 không khớp với các giá trị trên
                data[header_text] = content

    return data

# Đọc file JSON với danh sách các link đã cào trước đó
with open('../json/medicine.json', 'r', encoding='utf-8') as f:
    links_data = json.load(f)

# Danh sách chứa dữ liệu của tất cả các trang
all_details = []

# Duyệt qua tất cả các link để cào dữ liệu chi tiết
for item in links_data:
    link = item['link']
    print(f"Đang cào dữ liệu từ link: {link}")
    try:
        details = scrape_details(link)  # Cào dữ liệu từ link
        all_details.append(details)
    except Exception as e:
        print(f"Lỗi khi cào dữ liệu từ {link}: {str(e)}")

# Chuyển dữ liệu thành DataFrame và lưu thành file CSV
df = pd.DataFrame(all_details)
df.to_csv('../csv/medicine_details.csv', index=False, encoding='utf-8-sig')

print("Đã lưu dữ liệu vào file drug_details.csv.")
