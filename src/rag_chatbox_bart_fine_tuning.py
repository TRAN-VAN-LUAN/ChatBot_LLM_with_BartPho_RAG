# -*- coding: utf-8 -*-
"""Vietnamese Medical Q&A Chatbot using BARTpho"""

import os
import pickle
import logging
import pandas as pd
import torch
from typing import List
from transformers import PreTrainedTokenizerFast, AutoModelForQuestionAnswering
from langchain.vectorstores import LanceDB
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.schema import Document
from langchain.embeddings import SentenceTransformerEmbeddings
from underthesea import word_tokenize

# Setup logging configuration
logging.basicConfig(level=logging.INFO)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Constants for model and tokenizer paths
MODEL_NAME = 'vinai/bartpho-word-base'
EMBEDDING_MODEL = "keepitreal/vietnamese-sbert"
MODEL_PATH = 'E:/university/TLCN/ChatBox/model/fine_tuned_bartpho1'
TOKENIZER_PATH = 'E:/university/TLCN/ChatBox/model/fine_tuned_bartpho1'

def load_model_and_tokenizer(model_path: str, tokenizer_path: str):
    """Load the fine-tuned BARTpho model and tokenizer."""
    try:
        model = AutoModelForQuestionAnswering.from_pretrained(model_path).to(device)
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
        logging.info("Fine-tuned model and tokenizer loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading model or tokenizer: {e}")
        raise e

    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.save_pretrained(tokenizer_path)
        logging.info("Padding token added to tokenizer.")

    return model, tokenizer

model, tokenizer = load_model_and_tokenizer(MODEL_PATH, TOKENIZER_PATH)

def load_data(msd_file: str, tamanh_file: str) -> pd.DataFrame:
    """Load and preprocess medical question data from CSV files."""
    df_msd = pd.read_csv(msd_file)
    df_tamanh = pd.read_csv(tamanh_file)
    
    # Select and rename relevant columns
    df_msd_selected = df_msd[['Question', 'Detailed Content']]
    df_tamanh_selected = df_tamanh[['Question', 'Context']].rename(columns={'Context': 'Detailed Content'})
    
    # Concatenate data and clean underscores
    df_combined = pd.concat([df_msd_selected, df_tamanh_selected], ignore_index=True)
    df_combined['Question'] = df_combined['Question'].str.replace('_', ' ')
    df_combined['Detailed Content'] = df_combined['Detailed Content'].str.replace('_', ' ')
    
    return df_combined

df = load_data(
    'E:/university/TLCN/ChatBox/data/csv/processed_medical_full.csv', 
    'E:/university/TLCN/ChatBox/data/csv/processed_tamanh_hospital_cleaned_full.csv'
)

def save_docsearch(docsearch, filename='E:/university/TLCN/ChatBox/data/kl/docsearch_cache.pkl'):
    """Save the document search index to a file."""
    with open(filename, 'wb') as f:
        pickle.dump(docsearch, f)

def load_docsearch(filename='E:/university/TLCN/ChatBox/data/pkl/docsearch_cache.pkl'):
    """Load the document search index from a file."""
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    return None

def initialize_retrievers(df):
    """Initialize the retrievers for document search."""
    model_embedding = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
    documents = [Document(page_content=question, metadata={"source": index}) for index, question in enumerate(df['Question'])]

    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 1

    docsearch = load_docsearch()
    if docsearch is None:
        db = LanceDB.connect('/tmp/lancedb')
        docsearch = LanceDB.from_documents(documents, model_embedding, connection=db)
        save_docsearch(docsearch)

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, docsearch.as_retriever(search_kwargs={"k": 1})],
        weights=[0.4, 0.6]
    )
    
    return ensemble_retriever

ensemble_retriever = initialize_retrievers(df)

def get_details(df, page_contents):
    """Retrieve detailed content based on matched questions."""
    details = []
    for question in page_contents:
        row = df[df['Question'] == question]
        if not row.empty:
            detail = row['Detailed Content'].values[0]
            details.append(detail)
    return details

def remove_duplicates(segmented_contexts):
    """Remove duplicate words from segmented contexts."""
    return [list(set(context)) for context in segmented_contexts]

def segment_contexts(details):
    """Tokenize and segment context details into words."""
    segmented_contexts = []
    for context in details:
        segmented_text = word_tokenize(context)
        words_only = [word for word in segmented_text if word.strip()]
        segmented_contexts.append(words_only)
    return remove_duplicates(segmented_contexts)

def add_new_tokens_to_tokenizer(segmented_contexts, tokenizer):
    """Add new tokens from segmented contexts to the tokenizer."""
    new_tokens = [word for context_words in segmented_contexts for word in context_words]
    unique_new_tokens = list(set(new_tokens))
    tokenizer.add_tokens(unique_new_tokens)
    model.resize_token_embeddings(len(tokenizer))

# Sample question for testing
test_question = "Sau chuyển phôi đau bụng dưới bên phải có bất thường không?"
docs = ensemble_retriever.get_relevant_documents(test_question)
page_contents = [doc.page_content for doc in docs]
details = get_details(df, page_contents)
segmented_contexts = segment_contexts(details)
add_new_tokens_to_tokenizer(segmented_contexts, tokenizer)

def generate_answer(test_question: str, details: List[str], model, tokenizer):
    """Generate answers to a question based on context details."""
    answers = []
    for detail in details:
        inputs = tokenizer([test_question], [detail], return_tensors='pt', padding=True, truncation=True, max_length=512)
        inputs = {key: value.to(device) for key, value in inputs.items() if key != 'token_type_ids'}

        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)

        start_probs = torch.softmax(outputs.start_logits, dim=1)
        end_probs = torch.softmax(outputs.end_logits, dim=1)

        start_idx = torch.argmax(start_probs, dim=1).item()
        end_idx = torch.argmax(end_probs, dim=1).item()

        answer_tokens = inputs['input_ids'][0][start_idx:end_idx + 1]
        answer = tokenizer.convert_ids_to_tokens(answer_tokens.tolist())
        formatted_answer = ''.join([token + '</w>' if not token.endswith('</w>') else token for token in answer])
        formatted_answer = formatted_answer.replace('</w>', ' ')
        
        answers.append(formatted_answer)

    return answers

# Generate answers for the test question
answers = generate_answer(test_question, details, model, tokenizer)

# Print the generated answers
for i, answer in enumerate(answers):
    print(f"Answer {i+1}: {answer}")
