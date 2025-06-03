# -*- coding: utf-8 -*-
"""
GrapRAG - Graph-based Retrieval Augmented Generation for medical Q&A
"""

import re
import pandas as pd
import ast
import torch
from tqdm import tqdm
from collections import defaultdict
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from neo4j_graphrag.retrievers import VectorRetriever
from typing import List

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import os
os.environ["OPENAI_API_KEY"] = ""


# Load context data from CSV files
import os

# Lấy đường dẫn tương đối từ vị trí file hiện tại
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(current_dir)  # Thư mục gốc của dự án (lùi lại 1 cấp từ src)
csv_dir = os.path.join(base_dir, 'data', 'csv')

# Đọc các file CSV với đường dẫn tương đối
df_msd_context = pd.read_csv(os.path.join(csv_dir, 'msd_context_data.csv'))
df_vinmec_context = pd.read_csv(os.path.join(csv_dir, 'vinmec_context.csv'))
df_vinmec_qa = pd.read_csv(os.path.join(csv_dir, 'vinmec_qa.csv'))

df_context = pd.concat([df_msd_context, df_vinmec_context], ignore_index=True)

# Stopwords for preprocessing
multi_word_stopwords = ['dấu hiệu']
single_word_stopwords = set([
    'là', 'của', 'và', 'hoặc', 'có', 'trong',
    'các', 'những', 'đó', 'đây', 'ra', 'gì', 'theo',
    'để', 'cho', 'bệnh', 'thuốc'
])

# Initialize embedding model
model_emb = SentenceTransformer('keepitreal/vietnamese-sbert')


class CustomEmbedder:
    """Wrapper for SentenceTransformer model to work with VectorRetriever"""
    def __init__(self, model: SentenceTransformer):
        self.model = model

    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

def remove_stopwords(phrase):
    """Remove stopwords from a phrase"""
    phrase_lower = phrase.lower()

    # Remove multi-word stopwords
    for mw in multi_word_stopwords:
        pattern = r'\b' + re.escape(mw) + r'\b'
        phrase_lower = re.sub(pattern, '', phrase_lower)

    # Remove extra whitespace
    phrase_lower = re.sub(r'\s+', ' ', phrase_lower).strip()

    # Remove punctuation and split into tokens
    phrase_clean = re.sub(r'[^\w\s]', '', phrase_lower)
    tokens = phrase_clean.split()

    # Filter single stopwords
    filtered_tokens = [t for t in tokens if t not in single_word_stopwords]

    if not filtered_tokens:
        return ''

    # Restore original words from the original phrase
    original_words = phrase.split()
    filtered_original = []
    for w in original_words:
        w_clean = re.sub(r'[^\w\s]', '', w.lower())
        if w_clean in filtered_tokens:
            filtered_original.append(w)
            filtered_tokens.remove(w_clean)

    return ' '.join(filtered_original)

def sentence_to_keywords(text):
    """Convert a sentence to a list of keywords"""
    results = []
    seen = set()

    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    for sentence in sentences:
        if not sentence:
            continue

        phrases = re.split(r',(?![^(]*\))', sentence)
        phrases = [p.strip() for p in phrases if p.strip()]
        filtered_phrases = []

        for phrase in phrases:
            cleaned = remove_stopwords(phrase)
            if cleaned and cleaned not in seen:
                filtered_phrases.append(cleaned)

        max_len = len(filtered_phrases)

        if max_len == 1:
            # If only 1 valid phrase, add it
            single = filtered_phrases[0]
            if single not in seen:
                seen.add(single)
                results.append(single)

        elif max_len > 1:
            # Combine 2 to N phrases
            for n in range(2, max_len + 1):
                for i in range(max_len - n + 1):
                    merged = ', '.join(filtered_phrases[i:i + n])
                    if merged not in seen:
                        seen.add(merged)
                        results.append(merged)

    return results

def embed_nodes_and_create_vector_index(driver, labels, embedding_model, index_name, dim=768):
    """
    Sinh embedding lại toàn bộ node có label được chỉ định (trừ Chunk/Document)
    và chỉ xử lý node có embedding bị sai chiều hoặc chưa có embedding.
    """

    exclude_labels = {"Chunk", "Document"}

    with driver.session() as session:
        for label in labels:
            if label.lower() in exclude_labels:
                continue

            # Lấy node có id, và hoặc embedding bị thiếu hoặc sai chiều (khác dim)
            result = session.run(f"""
                MATCH (n:`{label}`)
                WHERE n.id IS NOT NULL AND (
                    n.embedding IS NULL OR size(n.embedding) <> $dim
                )
                RETURN id(n) AS node_id, n.id AS text
            """, dim=dim)

            nodes = [(record["node_id"], record["text"]) for record in result]

            if not nodes:
                continue

            # Sinh embedding
            texts = [text for _, text in nodes]
            embeddings = embedding_model.encode(texts, normalize_embeddings=True).tolist()

            # Cập nhật embedding mới vào graph
            for (node_id, _), embedding in tqdm(zip(nodes, embeddings), total=len(nodes), desc=f"→ Updating {label}"):
                session.run("""
                    MATCH (n)
                    WHERE id(n) = $node_id
                    SET n.embedding = $embedding
                    SET n:`__Entity__`
                """, node_id=node_id, embedding=embedding)

        # Xoá index cũ nếu có
        session.run(f"DROP INDEX {index_name} IF EXISTS")

        # Tạo index mới
        session.run(f"""
            CALL db.index.vector.createNodeIndex(
                $index_name,
                '__Entity__',
                'embedding',
                $dim,
                'cosine'
            )
        """, index_name=index_name, dim=dim)

# Check if index exists and create it if not
def initialize_driver_retriever(entity_group = 'BỆNH'):
    # Initialize retriever with error handling
    embedder = CustomEmbedder(model_emb)

    if entity_group == 'BỆNH':
        driver_benh = GraphDatabase.driver("neo4j+s://4881e1c6.databases.neo4j.io", auth=("neo4j", "D4KxdYEfgvRvDAsfvG71Hd0w9bNEyUo8N5s3EmjDmAk"))

        ## embedding các node bệnh
        labels_benh = ['Chủ đề', 'Tiêu đề', 'Nội dung', 'BỆNH', 'THUỘC_TÍNH_BỆNH', 'NỘI_DUNG_THUỘC_TÍNH_BỆNH']
        embed_nodes_and_create_vector_index(driver_benh, labels_benh, model_emb, 'benh_embedding', dim=768)
        retriever_benh = VectorRetriever(driver_benh, 'benh_embedding', embedder)
        return retriever_benh, driver_benh
    else:
        driver_thuoc = GraphDatabase.driver("neo4j+s://fb4f9b28.databases.neo4j.io", auth=("neo4j", "qJ2NxeAu5m8lcPtDD3njUaKRD6e9hExbw5GgCU5BxcE"))

        ## embedding các node thuốc
        labels_thuoc = ['THUỐC', 'THUỘC_TÍNH_THUỐC', 'NỘI_DUNG_THUỘC_TÍNH_THUỐC']
        embed_nodes_and_create_vector_index(driver_thuoc, labels_thuoc, model_emb, 'thuoc_embedding', dim=768)
        retriever_thuoc = VectorRetriever(driver_thuoc, 'thuoc_embedding', embedder)
        return retriever_thuoc, driver_thuoc

def search_queries(retriever, query_list, top_k=5, verbose=False):
    """
    Search for nodes matching each query in the list
    
    Args:
        retriever: Vector retriever object
        query_list: List of query texts
        top_k: Number of results to return per query
        verbose: Whether to print detailed results
        
    Returns:
        Tuple of (results_dict, high_score_results)
    """

    results_dict = {}
    high_score_results = []

    for query_text in query_list:
        results = retriever.search(query_text=query_text, top_k=top_k)
        results_dict[query_text] = results.items

        for item in results.items:
            score = item.metadata.get('score', 0)
            if score > 0.9:
                high_score_results.append(item)

    return results_dict, high_score_results

def get_existing_labels(driver):
    """Trả về tập hợp tất cả label hiện có trong DB."""
    with driver.session() as session:
        result = session.run("CALL db.labels()")
        return {record["label"] for record in result}

def find_entities_from_retriever_results(driver, high_score_results, target_field='id'):
    """
    Tìm các thực thể BỆNH và THUỐC từ kết quả Retriever.
    Trả về danh sách [(id, count)] đã sort, KHÔNG có prefix.
    """
    match_count = defaultdict(int)

    # Lấy các label thực tế có trong DB
    existing_labels = get_existing_labels(driver)

    # Chỉ giữ lại những entity_type có tồn tại trong DB
    candidate_entity_types = ['BỆNH', 'THUỐC', 'Chủ đề']
    valid_entity_types = [et for et in candidate_entity_types if et in existing_labels]

    with driver.session() as session:
        for item in high_score_results:
            try:
                content_data = ast.literal_eval(item.content)
            except (ValueError, SyntaxError):
                continue

            node_id = content_data.get('id')
            labels = item.metadata.get('nodeLabels', [])
            target_labels = [label for label in labels if label != '__Entity__']

            if not node_id or not target_labels:
                continue

            for target_label in target_labels:
                for entity_type in valid_entity_types:
                    query = f"""
                    MATCH (e:`{entity_type}`)-[*1..2]->(n:`{target_label}`)
                    WHERE toLower(n.{target_field}) = toLower($value)
                    RETURN e.id AS id
                    UNION
                    MATCH (e:`{entity_type}`)
                    WHERE toLower(e.{target_field}) = toLower($value)
                    RETURN e.id AS id
                    """
                    result = session.run(query, value=node_id)
                    for record in result:
                        match_count[record['id']] += 1

    return sorted(match_count.items(), key=lambda x: x[1], reverse=True)

def filter_question(question, model_name="gpt-4o-mini"):
    llm = ChatOpenAI(model_name=model_name, temperature=0)
    response = llm([
        SystemMessage(content="Xác định xem câu hỏi sau liên quan đến THUỐC hay BỆNH hoặc khác. Chỉ cần trả lời THUỐC hoặc BỆNH"),
        HumanMessage(content=question)
    ])
    return response.content.strip()

def get_top_topic_contexts(retriever, driver, question, df_context):
    results_dict, high_score_results = search_queries(
        retriever,
        sentence_to_keywords(question)
    )
    result_find = find_entities_from_retriever_results(
        driver,
        high_score_results
    )

    if not result_find:
        return []

    # Tìm giá trị count cao nhất
    max_count = max(result_find, key=lambda x: x[1])[1]

    # Lấy tất cả topic có count bằng max_count
    top_topics = [topic for topic, count in result_find if count == max_count]

    # Lấy context tương ứng với các topic này
    context_rows = df_context[df_context["topic"].isin(top_topics)]
    contexts = context_rows["context"].tolist()

    # Tạo danh sách Document
    documents = [
        Document(page_content=context, metadata={"source": idx})
        for idx, context in enumerate(contexts)
    ]

    # Tách văn bản thành các chunk nhỏ
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=712,
        chunk_overlap=0,
        separators=["."]
    )
    texts = text_splitter.split_documents(documents)

    return [doc.page_content for doc in texts]


def get_context_from_question(question):
    """
    Simple function that takes a question and returns relevant contexts
    
    Args:
        question: The user's question
        
    Returns:
        List of context strings
    """
    # Check for empty question
    if not question or question.strip() == "":
        return []
        
    try:
        retriever, driver = initialize_driver_retriever(filter_question(question))
        if not retriever or not driver:
            print("Failed to initialize retriever or driver.")
            return []
        # Get top topic contexts based on the question
        contexts = get_top_topic_contexts(retriever, driver, question, df_context)
        return contexts
    except Exception as e:
        print(f"Error during context retrieval: {e}")
        return []

# Only run this when the script is executed directly (not when imported)
if __name__ == "__main__":
    import time
    start_time = time.time()
    
    elapsed_time = time.time() - start_time
    print(f"Process completed in {elapsed_time:.2f} seconds")



