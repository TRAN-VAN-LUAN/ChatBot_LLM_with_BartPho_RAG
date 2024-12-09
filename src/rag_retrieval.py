# -*- coding: utf-8 -*-
"""Retrieval Module for Vietnamese Medical Q&A Chatbot using Langchain and Sentence Transformers"""

import os
import pickle
import dill
import logging
import pandas as pd
import torch
import io
from langchain.vectorstores import LanceDB
import lancedb
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.schema import Document
from langchain.embeddings import SentenceTransformerEmbeddings
from transformers.models.roberta.modeling_roberta import RobertaSelfAttention
from underthesea import word_tokenize
from langchain_experimental.text_splitter import SemanticChunker

# Setup logging configuration
logging.basicConfig(level=logging.INFO)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Constants for model and tokenizer paths
EMBEDDING_MODEL = "keepitreal/vietnamese-sbert"
# Create embeddings using the SentenceTransformerEmbeddings class
model_embedding = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)

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

# Lớp tùy chỉnh để tải dữ liệu trên CPU và xử lý ánh xạ lớp
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Chuyển hướng việc tải các tensor PyTorch để đảm bảo chúng được tải trên CPU
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location=torch.device('cpu'))
        
        # Ánh xạ lớp 'RobertaSdpaSelfAttention' sang 'RobertaSelfAttention'
        if name == 'RobertaSdpaSelfAttention':
            return RobertaSelfAttention

        return super().find_class(module, name)

    
# Hàm để tải docsearch từ file pickle
def load_docsearch(filename='E:/university/TLCN/ChatBox/data/lancedb'):
    # If model_embedding is provided, use it for initializing the connection
    if model_embedding is None:
        raise ValueError("Model embedding must be provided.")
    
    db_retriever = lancedb.connect(filename)  # Connect to LanceDB database
    docsearch = LanceDB(connection=db_retriever, embedding=model_embedding)
    return docsearch

def initialize_retrievers(df):
    """Initialize the retrievers for document search."""
    
    # Create a list of documents from the questions
    documents = [Document(page_content=question, metadata={"source": index}) for index, question in enumerate(df['Question'])]
    
    # Initialize BM25 retriever
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 1
    
    # Load LanceDB document search
    docsearch = load_docsearch()
    logging.info('LanceDB loaded.')
    
    # Check if LanceDB has the as_retriever method
    try:
        doc_retriever = docsearch.as_retriever(search_kwargs={"k": 2})
    except AttributeError:
        logging.error("LanceDB instance does not have the `as_retriever` method. Please check the implementation.")
        return None

    # Initialize the EnsembleRetriever with both BM25 and LanceDB retrievers
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, doc_retriever],
        weights=[0.4, 0.6]
    )
    return ensemble_retriever

def get_details(df, page_contents):
    """Retrieve detailed content based on matched questions."""
    details = []
    for question in page_contents:
        row = df[df['Question'] == question]
        if not row.empty:
            detail = row['Detailed Content'].values[0]
            details.append(detail)
    return details

def semantic_chunks(details):
    """"""
    semantic_chunker = SemanticChunker(model_embedding, breakpoint_threshold_type="percentile")
    semantic_chunks = semantic_chunker.create_documents([d for d in details])
    return semantic_chunks

def segment_contexts(semantic_chunks):
    """Tokenize and segment context details into words."""
    segmented_contexts = []
    for context in semantic_chunks:
        segmented_text = word_tokenize(context)
        words_only = [word for word in segmented_text if word.strip()]
        segmented_contexts.append(words_only)
    return segmented_contexts

# Load data
df = load_data(
    'E:/university/TLCN/ChatBox/data/csv/processed_medical_full.csv', 
    'E:/university/TLCN/ChatBox/data/csv/processed_tamanh_hospital_cleaned_full.csv'
)

# Initialize retrievers
ensemble_retriever = initialize_retrievers(df)

def retrieve_documents(test_question: str):
    """Retrieve documents based on the question."""
    
    # Kiểm tra xem ensemble_retriever có phải là None
    if ensemble_retriever is None:
        logging.error("Ensemble retriever không được khởi tạo.")
        return [], []
    
    # Thử truy xuất tài liệu
    try:
        docs = ensemble_retriever.invoke(test_question)  
        if not docs:
            logging.warning("Không có tài liệu nào được truy xuất.")
            return [], []

        page_contents = [doc.page_content for doc in docs]
        details = get_details(df, page_contents)
        # prompt = semantic_chunks(details)
        segmented_contexts = segment_contexts(details)

        return details, segmented_contexts
    
    except AttributeError as e:
        logging.error(f"Lỗi AttributeError trong quá trình truy xuất tài liệu: {e}")
        return [], []

# Example test call (make sure to uncomment for testing)
# details, segmented_contexts = retrieve_documents("What is the capital of France?")
