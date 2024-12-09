from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import json
import time

# Hàm đợi trang và click vào số trang
def wait_and_click_page(page_num):
    try:
        # Đợi cho phần tử trang xuất hiện (sử dụng id dựa trên vị trí số trang)
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, f"//a[text()='{page_num}']"))
        )

        # Kích hoạt sự kiện onclick thông qua JavaScript
        page_link = driver.find_element(By.XPATH, f"//a[text()='{page_num}']")
        driver.execute_script("arguments[0].click();", page_link)

        # Đợi trang tải xong (có thể thêm sleep nếu cần)
        time.sleep(3)
        print(f"Đã chuyển sang trang {page_num}")
    except Exception as e:
        print(f"Không thể chuyển sang trang {page_num}: {str(e)}")

# Hàm cào dữ liệu trong trang hiện tại
def scrape_page():
    page_data = []
    try:
        # Tìm phần tử 'ul' có class 'list_drug'
        ul_element = driver.find_element(By.CLASS_NAME, 'list_drug')

        # Lấy tất cả các thẻ 'li' bên trong thẻ 'ul'
        li_elements = ul_element.find_elements(By.TAG_NAME, 'li')

        # Duyệt qua tất cả các thẻ 'li'
        for li in li_elements:
            a_tag = li.find_element(By.TAG_NAME, 'a')
            link = a_tag.get_attribute('href')
            title = a_tag.text
            page_data.append({'title': title, 'link': link})
    except Exception as e:
        print(f"Lỗi khi cào dữ liệu: {str(e)}")
    return page_data

# Mở trình duyệt
driver = webdriver.Chrome()

# Truy cập trang đầu tiên
driver.get('https://www.vinmec.com/vie/thuoc/')

all_data = []
total_pages = 8  # Giả sử có 8 trang

for page_num in range(1, total_pages + 1):
    wait_and_click_page(page_num)  # Chuyển sang trang mới
    page_data = scrape_page()  # Cào dữ liệu từ trang hiện tại
    all_data.extend(page_data)  # Gộp dữ liệu từ các trang vào danh sách chung

# Lưu kết quả vào file JSON
with open('../json/medicine.json', 'w', encoding='utf-8') as f:
    json.dump(all_data, f, ensure_ascii=False, indent=4)

# Đóng trình duyệt
driver.quit()
