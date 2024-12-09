# -*- coding: utf-8 -*-
"""Generation Module for Vietnamese Medical Q&A Chatbot using BARTpho"""

import torch
from transformers import AutoModelForQuestionAnswering, PreTrainedTokenizerFast
from rag_retrieval import retrieve_documents  # Import the retrieve_documents function
from underthesea import word_tokenize
import gradio as gr
import logging

# Setup logging configuration
logging.basicConfig(level=logging.INFO)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = 'E:/university/TLCN/ChatBox/model/fine_tuned_bartpho1'
TOKENIZER_PATH = 'E:/university/TLCN/ChatBox/model/fine_tuned_bartpho1'

def load_model_and_tokenizer(model_path: str, tokenizer_path: str):
    """Load the fine-tuned BARTpho model and tokenizer."""
    try:
        # Load the model and ensure it is loaded on CPU
        model = AutoModelForQuestionAnswering.from_pretrained(model_path)
        
        # Move model to CPU (if not already done)
        model.to(torch.device('cpu'))
        
        # Load the tokenizer
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
        logging.info("Fine-tuned model and tokenizer loaded successfully.")
        
    except RuntimeError as e:
        if "CUDA device" in str(e):
            logging.error("CUDA device error: Attempting to load model on a CUDA device while CUDA is unavailable. Ensure the model is compatible.")
            raise e
        else:
            logging.error(f"Error loading model or tokenizer: {e}")
            raise e
    except Exception as e:
        logging.error(f"Error loading model or tokenizer: {e}")
        raise e

    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.save_pretrained(tokenizer_path)
        logging.info("Padding token added to tokenizer.")

    return model, tokenizer

# Load model and tokenizer
model, tokenizer = load_model_and_tokenizer(MODEL_PATH, TOKENIZER_PATH)

def generate_answer(test_question: str, details):
    """Generate answers to a question based on context details."""
    # Tokenize câu hỏi và bối cảnh
    answers = []
    combined_answer = ""
    for i, detail in enumerate(details):
        inputs = tokenizer(test_question, detail, return_tensors='pt', padding=True, truncation=True, max_length=512)
        inputs = {key: value.to(device) for key, value in inputs.items() if key not in ['token_type_ids']}

        # Sinh ra dự đoán từ mô hình
        model.to(device)
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)

        # Lấy logits dự đoán vị trí bắt đầu và kết thúc
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        # Chuyển đổi logits thành xác suất
        start_probs = torch.softmax(start_logits, dim=1)
        end_probs = torch.softmax(end_logits, dim=1)

        # Lấy vị trí có xác suất cao nhất
        start_idx = torch.argmax(start_probs, dim=1).item()
        end_idx = torch.argmax(end_probs, dim=1).item()

        # Trích xuất câu trả lời từ context dựa trên vị trí dự đoán
        answer_tokens = inputs['input_ids'][0][start_idx:end_idx + 1]
        answer = tokenizer.convert_ids_to_tokens(answer_tokens.tolist())  # Chuyển đổi ID thành token
        answer_with_w = [token + '</w>' if not token.endswith('</w>') else token for token in answer]
        answer_str = ''.join(answer_with_w)
        formatted_answer = answer_str.replace('</w>', ' ')
        answers.append(formatted_answer)
        if i > 0 and formatted_answer:
            combined_answer += ". "
        combined_answer += formatted_answer
    return combined_answer

# Gradio interface
def answer_question(test_question: str):
    """Process the question and generate an answer."""
    details, segmented_contexts = retrieve_documents(test_question)  # Retrieve relevant details
    answers = generate_answer(test_question, details)  # Generate answers based on retrieved details
    return answers

# Create Gradio interface
def run_gradio():
    interface = gr.Interface(
        fn=answer_question,
        inputs="text",  # Input question
        outputs="text",  # Output answer
        title="Vietnamese Medical Q&A Chatbot",
        description="Nhập câu hỏi và mô hình sẽ trả lời dựa trên ngữ cảnh."
    )
    
    interface.launch()

# Run the Gradio application
if __name__ == "__main__":
    run_gradio()
