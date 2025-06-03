import pandas as pd
from docx import Document

def extract_drug_info(docx_file):
    # Mở tài liệu Word
    doc = Document(docx_file)
    
    drug_data = []
    current_drug = {}
    current_section = None  # Biến để theo dõi phần hiện tại

    for para in doc.paragraphs:
        text = para.text.strip()

        # Kiểm tra nếu đoạn văn là tên thuốc
        if text.startswith("Tên chung quốc tế:"):
            if current_drug:  # Lưu thuốc trước đó nếu có
                drug_data.append(current_drug)
                current_drug = {}

            current_drug['Tên chung quốc tế'] = text.split(":", 1)[1].strip()
            current_section = 'Tên chung quốc tế'  # Cập nhật phần hiện tại
            
        elif text.startswith("Mã ATC:") and current_drug:
            current_drug['Mã ATC'] = text.split(":", 1)[1].strip()
            current_section = 'Mã ATC'
        elif text.startswith("Loại thuốc:") and current_drug:
            current_drug['Loại thuốc'] = text.split(":", 1)[1].strip()
            current_section = 'Loại thuốc'
        elif text.startswith("Dạng thuốc và hàm lượng") and current_drug:
            current_drug['Dạng thuốc và hàm lượng'] = []
            current_section = 'Dạng thuốc và hàm lượng'
        elif current_section == 'Dạng thuốc và hàm lượng' and text and not text.startswith("Dược lực học"):
            current_drug['Dạng thuốc và hàm lượng'].append(text)
        elif text.startswith("Dược lực học") and current_drug:
            current_drug['Dược lực học'] = []
            current_section = 'Dược lực học'
        elif current_section == 'Dược lực học' and text and not text.startswith("Dược động học"):
            current_drug['Dược lực học'].append(text)
        elif text.startswith("Dược động học") and current_drug:
            current_drug['Dược động học'] = []
            current_section = 'Dược động học'
        elif current_section == 'Dược động học' and text and not text.startswith("Chỉ định"):
            current_drug['Dược động học'].append(text)
        elif text.startswith("Chỉ định") and current_drug:
            current_drug['Chỉ định'] = []
            current_section = 'Chỉ định'
        elif current_section == 'Chỉ định' and text and not text.startswith("Chống chỉ định"):
            current_drug['Chỉ định'].append(text)
        elif text.startswith("Chống chỉ định") and current_drug:
            current_drug['Chống chỉ định'] = []
            current_section = 'Chống chỉ định'
        elif current_section == 'Chống chỉ định' and text and not text.startswith("Thận trọng"):
            current_drug['Chống chỉ định'].append(text)
        elif text.startswith("Thận trọng") and current_drug:
            current_drug['Thận trọng'] = []
            current_section = 'Thận trọng'
        elif current_section == 'Thận trọng' and text and not text.startswith("Thời kỳ mang thai"):
            current_drug['Thận trọng'].append(text)
        elif text.startswith("Thời kỳ mang thai") and current_drug:
            current_drug['Thời kỳ mang thai'] = []
            current_section = 'Thời kỳ mang thai'
        elif current_section == 'Thời kỳ mang thai' and text and not text.startswith("Thời kỳ cho con bú"):
            current_drug['Thời kỳ mang thai'].append(text)
        elif text.startswith("Thời kỳ cho con bú") and current_drug:
            current_drug['Thời kỳ cho con bú'] = []
            current_section = 'Thời kỳ cho con bú'
        elif current_section == 'Thời kỳ cho con bú' and text and not text.startswith("Tác dụng không mong muốn (ADR)"):
            current_drug['Thời kỳ cho con bú'].append(text)
        elif text.startswith("Tác dụng không mong muốn (ADR)") and current_drug:
            current_drug['Tác dụng không mong muốn (ADR)'] = []
            current_section = 'Tác dụng không mong muốn (ADR)'
        elif current_section == 'Tác dụng không mong muốn (ADR)' and text and not text.startswith("Hướng dẫn cách xử trí ADR"):
            current_drug['Tác dụng không mong muốn (ADR)'].append(text)
        elif text.startswith("Hướng dẫn cách xử trí ADR") and current_drug:
            current_drug['Hướng dẫn cách xử trí ADR'] = []
            current_section = 'Hướng dẫn cách xử trí ADR'
        elif current_section == 'Hướng dẫn cách xử trí ADR' and text and not text.startswith("Liều lượng và cách dùng"):
            current_drug['Hướng dẫn cách xử trí ADR'].append(text)
        elif text.startswith("Liều lượng và cách dùng") and current_drug:
            current_drug['Liều lượng và cách dùng'] = []
            current_section = 'Liều lượng và cách dùng'
        elif current_section == 'Liều lượng và cách dùng' and text and not text.startswith("Cách dùng"):
            current_drug['Liều lượng và cách dùng'].append(text)
        elif text.startswith("Cách dùng") and current_drug:
            current_drug['Cách dùng'] = []
            current_section = 'Cách dùng'
        elif current_section == 'Cách dùng' and text and not text.startswith("Liều dùng"):
            current_drug['Cách dùng'].append(text)
        elif text.startswith("Liều dùng") and current_drug:
            current_drug['Liều dùng'] = []
            current_section = 'Liều dùng'
        elif current_section == 'Liều dùng' and text and not text.startswith("Tương tác thuốc"):
            current_drug['Liều dùng'].append(text)
        elif text.startswith("Tương tác thuốc") and current_drug:
            current_drug['Tương tác thuốc'] = []
            current_section = 'Tương tác thuốc'
        elif current_section == 'Tương tác thuốc' and text and not text.startswith("Quá liều và xử trí"):
            current_drug['Tương tác thuốc'].append(text)
        elif text.startswith("Quá liều và xử trí") and current_drug:
            current_drug['Quá liều và xử trí'] = []
            current_section = 'Quá liều và xử trí'
        elif current_section == 'Quá liều và xử trí' and text and not text.startswith("Cập nhật lần cuối"):
            current_drug['Quá liều và xử trí'].append(text)

    # Thêm thuốc cuối cùng vào danh sách nếu có
    if current_drug:
        drug_data.append(current_drug)

    return drug_data

# Đường dẫn đến tệp docx của bạn
docx_files = ['../docx/medicine1.docx', '../docx/medicine2.docx']

# Khởi tạo danh sách để chứa thông tin thuốc từ tất cả các tệp
all_drug_info = []

# Cào dữ liệu từ từng tệp docx
for docx_file in docx_files:
    drug_info = extract_drug_info(docx_file)
    all_drug_info.extend(drug_info)  # Kết hợp dữ liệu từ các tệp

# Chuyển đổi dữ liệu thành DataFrame và lưu vào tệp CSV
df = pd.DataFrame(all_drug_info)
csv_file = '../csv/medical_bank.csv'
df.to_csv(csv_file, index=False, encoding='utf-8-sig')

print(f'Dữ liệu đã được lưu vào {csv_file}')
