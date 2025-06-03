# -*- coding: utf-8 -*-
"""Generation Module for Vietnamese Medical Q&A Chatbot using LLaMA with Vinmec dataset"""

import torch
import re
import numpy as np
from transformers import PreTrainedTokenizerFast
from graprag import get_context_from_question  # Import the get_context_from_question function from graprag.py
from underthesea import word_tokenize
import gradio as gr 
import logging
import onnxruntime as ort

# Setup logging configuration
logging.basicConfig(level=logging.INFO)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Biến toàn cục để lưu trữ model và tokenizer
model = None
tokenizer = None

import os

# Sử dụng đường dẫn tương đối một cách đúng đắn
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(current_dir)  # Thư mục gốc của dự án
model_dir = os.path.join(base_dir, 'model', 'bartpho')

ONNX_MODEL_PATH = os.path.join(model_dir, 'bartpho_qa.onnx')
TOKENIZER_PATH = model_dir

def load_session_and_tokenizer(onnx_model_path: str, tokenizer_path: str):
    """Load the fine-tuned BARTpho model and tokenizer."""
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path, local_files_only=True)
    model = ort.InferenceSession(onnx_model_path)

    return model, tokenizer

# Khởi tạo model và tokenizer khi cần thiết
def get_session_and_tokenizer():
    global session, tokenizer
    if model is None or tokenizer is None:
        logging.info("Đang tải model và tokenizer...")
        session, tokenizer = load_session_and_tokenizer(ONNX_MODEL_PATH, TOKENIZER_PATH)
    return session, tokenizer

def generate_answer(question: str, contexts: list) -> str:
    answers = []
    
    # Chỉ lấy tối đa 2 context đầu tiên
    contexts_to_process = contexts[:2] if len(contexts) > 2 else contexts

    for context in contexts_to_process:
        # Xử lý context: thay _ bằng khoảng trắng
        context = context.replace('_', ' ')

        # Tokenize đầu vào
        inputs = tokenizer(question, context, return_tensors="pt")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Chuyển sang numpy cho onnxruntime
        onnx_inputs = {
            "input_ids": input_ids.cpu().numpy(),
            "attention_mask": attention_mask.cpu().numpy()
        }

        # Chạy inference với ONNX
        start_logits, end_logits = session.run(["start_logits", "end_logits"], onnx_inputs)

        # Lấy vị trí start và end token có xác suất cao nhất
        start_idx = np.argmax(start_logits[0])
        end_idx = np.argmax(end_logits[0]) + 1  # +1 để lấy token cuối cùng        # Giải mã token thành câu trả lời
        answer_ids = input_ids[0][start_idx:end_idx]
        answer = tokenizer.decode(answer_ids, skip_special_tokens=True)
        
        # Loại bỏ thủ công các thẻ <s> và </s> nếu còn sót lại
        answer = answer.replace("<s>", "").replace("</s>", "")

        answers.append(answer.strip())

    # Kết hợp các câu trả lời thành một chuỗi
    combined_answer = ". ".join(filter(None, answers))
    # Đảm bảo câu trả lời kết hợp không rỗng
    if not combined_answer:
        combined_answer = "Không tìm thấy câu trả lời phù hợp"
        
    return combined_answer


def remove_whitespace_around_dot(text):
    # Tìm tất cả các dấu "." và loại bỏ khoảng trắng trước và sau dấu "."
    return re.sub(r'\s*\.\s*', '.', text)

def answer_question(test_question: str, history):
    """Process the question, generate an answer, and update conversation history."""
    
    # Initialize history if it's None
    if history is None:
        history = []
    logging.info('_'*50)
    logging.info(f'question: {test_question}')

    clean_question = remove_whitespace_around_dot(test_question)
    
    # Use both retrieval methods: original and get_context_from_question
    graph_contexts = get_context_from_question(clean_question)
    
      # Generate an answer based on the retrieved details
    print(f"graph_contexts: {graph_contexts}")
    if graph_contexts:
        answer = generate_answer(clean_question, graph_contexts)
    else:
        answer = 'Không tìm thấy câu trả lời phù hợp!'
        
    # Kiểm tra câu trả lời trống hoặc chỉ chứa thẻ đặc biệt
    if not answer or answer.strip() in ['<s>', '</s>', '<s></s>'] or answer.isspace():
        answer = 'Không tìm thấy câu trả lời phù hợp!'    # Chuyển đổi các thẻ <br> thành xuống dòng thực sự
    answer = answer.replace("<br>", "\n")
    
    # Format the answer for better readability
    processed_parts = []
    for part in answer.split(', '):
        if ':' in part:
            key, value = part.rsplit(':', 1)  # Dùng rsplit để chỉ tách dấu `:` cuối cùng
            if value.strip():  # Nếu sau dấu ':' không rỗng
                processed_parts.append(f"{key.strip()} : {value.strip()}")
        else:
            processed_parts.append(part.strip())

    # Gộp lại thành chuỗi với dấu xuống dòng
    answer = '\n'.join(processed_parts)
    logging.info(f'answer: {answer}')

    # Update the conversation history with question and answer
    history.append((test_question, answer))
    
    # Return updated conversation history
    return history, ""

def handle_submit(user_message, history):
    """Xử lý logic khi người dùng nhấn Enter hoặc Submit."""
    history = history if not isinstance(history, gr.State) else history.value  # Truy cập giá trị thực tế của đối tượng State
    updated_history, _ = answer_question(user_message, history)
    return updated_history, ""

# def handle_suggestion(suggestion):
#     """Xử lý khi người dùng chọn một gợi ý."""
#     return suggestion

def run_gradio():
    # Tải model và tokenizer ngay khi khởi động ứng dụng
    global session, tokenizer
    logging.info("Đang tải model và tokenizer trước khi khởi chạy giao diện...")
    session, tokenizer = load_session_and_tokenizer(ONNX_MODEL_PATH, TOKENIZER_PATH)
    logging.info("model và tokenizer đã được tải thành công!")

    # Suggested queries for the user to click (Vietnamese medical questions)
    suggestions = [
        "Các triệu chứng thường gặp của bệnh Covid-19 là gì?", 
        "Thuốc paracetamol dùng để làm gì?",
        "Triệu chứng của bệnh tiểu đường", 
        "Cách điều trị cao huyết áp", 
        "Dùng thuốc Amoxicillin như thế nào?", 
        "Vitamin K1 có tác dụng gì?", 
        "Tác dụng phụ của thuốc aspirin", 
        "Thế nào là nhồi máu cơ tim?",
        "Cách phòng bệnh sốt xuất huyết",
        "Cảnh báo khi sử dụng thuốc Panadol", 
        "Triệu chứng của bệnh viêm gan",
        "Thuốc Cetirizine có tác dụng gì?",
    ]    
    with gr.Blocks(css="""        .suggestion-container {
            background-color: #f9f9f9;
            border: 2px solid #ddd;
            padding: 15px;
            width: 20% !important;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
            margin-right: 20px;
            flex-wrap: nowrap !important; /* Cho phép cuộn nếu nội dung quá dài */
            height: 566px;
            overflow-x: auto;  /* Cho phép cuộn theo chiều ngang */
            overflow-y: auto;  /* Cho phép cuộn theo chiều dọc */
            flex-shrink: 0;    /* Không cho phép co lại */
            order: -1;         /* Hiển thị bên trái */
            flex: 2;           /* Chiếm 2 phần */
        }
                   
        #component-10 {
            overflow: visible !important;
            display: flex;
            justify-content: flex-start;
        }

        .suggestion-title {
            font-size: 16px;
            font-weight: bold;
            color: #333;
            text-align: left;
            margin-bottom: 10px;
            overflow: hidden !important;
        }

        .suggestion-buttons {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        .suggestion-button {
            background-color: #f0f0f0;
            color: #333;
            font-size: 12px;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 8px 12px;
            cursor: pointer;
            transition: all 0.3s ease;
            width: 100%;
            text-align: left;
        }

        .suggestion-button:hover {
            background-color: #4CAF50;
            color: white;
            border-color: #4CAF50;
        }

        .suggestion-button:active {
            background-color: #45a049;
        }        .chat-container {
            display: flex;
            flex-direction: column;  /* Đặt chatbot và textbox theo chiều dọc */
            width: 80% !important;
            flex-grow: 1;            /* Để chatbot mở rộng chiếm phần còn lại */
            flex: 8;                 /* Chiếm 8 phần */
        }

        .chatbot-container {
            flex-grow: 1;  /* Đảm bảo chatbot chiếm phần còn lại */
        }
          /* Đảm bảo Row chính hiển thị chính xác với tỷ lệ 2:8 */
        .gradio-row {
            display: flex;
            flex-direction: row;
            align-items: stretch;
            width: 100%;
        }
    """) as demo:        # Header section
        gr.Markdown(
            """
            <h1 style='text-align:center; font-size: 50px; font-family: "Helvetica", sans-serif; 
                    color: #4CAF50; text-shadow: 2px 2px 10px rgba(0, 0, 0, 0.3); 
                    letter-spacing: 2px;'>
                Medical QA Chatbot
            </h1>
            <h2 style='text-align:center; font-size: 20px; font-family: "Helvetica", sans-serif; 
                    color: #555; margin-top: -15px;'>
                Sử dụng dữ liệu y tế Vinmec
            </h2>
            """
        )          # Create a State component for chat history
        chat_history = gr.State([])
          # Main content: Using gr.Row() to layout suggestions and chatbot side by side with ratio 2:8
        with gr.Row():
            # Suggestions section on the left (2 parts)
            with gr.Column(elem_classes=["suggestion-container"], scale=2):
                gr.Markdown("<div class='suggestion-title'>Gợi ý câu hỏi y tế</div>")
                # Create buttons for suggestions
                suggestion_buttons = []
                for suggestion in suggestions:
                    btn = gr.Button(suggestion, elem_classes=["suggestion-button"])
                    suggestion_buttons.append(btn)
            
            # Chat section (column for better alignment) (8 parts)
            with gr.Column(elem_classes=["chat-container"], scale=8):
                chatbot = gr.Chatbot(label="Lịch sử trò chuyện", elem_classes=["chatbot-container"])
                msg = gr.Textbox(label="Câu hỏi của bạn", placeholder="Hỏi về thuốc, bệnh, triệu chứng...", lines=1)
                submit_btn = gr.Button("Gửi")
                
        # Add click events for suggestion buttons after all UI components are created
        for btn in suggestion_buttons:
            btn.click(
                handle_submit, inputs=[btn, chat_history], outputs=[chatbot, msg], queue=False
            )
        msg.submit(handle_submit, [msg, chat_history], [chatbot, msg], queue=False)
        submit_btn.click(handle_submit, [msg, chat_history], [chatbot, msg], queue=False)    # Launch the Gradio interface
    # Bắt buộc sử dụng share=True vì môi trường không cho phép truy cập localhost
    logging.info("Khởi chạy với share=True")
    try:
        # Sử dụng inbrowser để tự động mở trình duyệt khi ứng dụng sẵn sàng
        demo.launch(share=True, inbrowser=True)
    except Exception as e:
        logging.error(f"Lỗi khởi chạy với share=True: {e}")
        try:
            # Thử với cách đơn giản nhất khi có lỗi
            logging.info("Thử khởi chạy đơn giản với share=True")
            demo.launch(share=True)
        except Exception as e2:
            logging.error(f"Không thể khởi chạy Gradio: {e2}")
            logging.info("Vui lòng kiểm tra kết nối mạng và thử lại.")
    
# Run the Gradio application
if __name__ == "__main__":
    run_gradio()