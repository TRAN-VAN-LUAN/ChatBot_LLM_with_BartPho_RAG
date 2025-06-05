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

# Stopwords for preprocessing
multi_word_stopwords = ['dấu hiệu', 'thế nào']
single_word_stopwords = set([
    'là', 'của', 'và', 'hoặc', 'có', 'trong',
    'các', 'những', 'đó', 'đây', 'ra', 'gì', 'theo',
    'để', 'cho', 'thuốc'
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
    phrase = re.sub(r'[?.!]+$', '', phrase).strip()

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

# Check if index exists and create it if not
def initialize_driver_retriever(entity_group = 'BỆNH'):
    # Initialize retriever with error handling
    embedder = CustomEmbedder(model_emb)

    if entity_group == 'BỆNH':
        driver_benh = GraphDatabase.driver("neo4j+s://4881e1c6.databases.neo4j.io", auth=("neo4j", "D4KxdYEfgvRvDAsfvG71Hd0w9bNEyUo8N5s3EmjDmAk"))
        retriever_benh = VectorRetriever(driver_benh, 'benh_embedding', embedder)
        return retriever_benh, driver_benh
    else:
        driver_thuoc = GraphDatabase.driver("neo4j+s://fb4f9b28.databases.neo4j.io", auth=("neo4j", "qJ2NxeAu5m8lcPtDD3njUaKRD6e9hExbw5GgCU5BxcE"))
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
    all_results = []

    # Truy vấn từng từ khóa
    for query_text in query_list:
        results = retriever.search(query_text=query_text, top_k=top_k)
        results_dict[query_text] = results.items
        all_results.extend(results.items)

    # Tìm ngưỡng phù hợp
    threshold = 0.7  # mặc định
    if any(item.metadata.get("score", 0) > 0.9 for item in all_results):
        threshold = 0.9
    elif any(item.metadata.get("score", 0) > 0.8 for item in all_results):
        threshold = 0.8

    # Lọc kết quả theo ngưỡng đã chọn
    high_score_results = [
        item for item in all_results
        if item.metadata.get("score", 0) > threshold
    ]

    return results_dict, high_score_results

def get_existing_labels(driver):
    """Trả về tập hợp tất cả label hiện có trong DB."""
    with driver.session() as session:
        result = session.run("CALL db.labels()")
        return {record["label"] for record in result}

def find_parent_node(session, node_id):
    """
    Tìm node cha của node nội dung.
    """
    query = """
    MATCH (parent)-[*1..2]->(child)
    WHERE child.id = $node_id
    RETURN parent.id AS parent_id, labels(parent) AS parent_labels
    LIMIT 1
    """
    result = session.run(query, node_id=node_id)
    record = result.single()
    if record:
        return record['parent_id'], record['parent_labels']
    return None, None

def find_entities_from_retriever_results(driver, high_score_results, target_field='id'):
    """
    Tìm các thực thể BỆNH và THUỐC từ kết quả Retriever.
    Trả về danh sách [(id, [list các node trùng])], đã sort theo độ dài list giảm dần.
    Nếu [list các node trùng] chứa các node là Nội dung, NỘI_DUNG_THUỘC_TÍNH_BỆNH, NỘI_DUNG_THUỘC_TÍNH_THUỐC thì tìm về node phía trước nó.
    """
    match_map = defaultdict(list)

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

            # Nếu là node nội dung, tìm node cha
            if any(label in ['Nội dung', 'NỘI_DUNG_THUỘC_TÍNH_BỆNH', 'NỘI_DUNG_THUỘC_TÍNH_THUỐC'] for label in target_labels):
                parent_id, parent_labels = find_parent_node(session, node_id)
                if parent_id and parent_labels:
                    node_id = parent_id
                    target_labels = [label for label in parent_labels if label != '__Entity__']

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
                        matched_id = record['id']
                        match_map[matched_id].append(node_id)  # lưu lại các node trùng với thực thể

    # Trả về danh sách (id, [list các node trùng]) được sort theo độ dài list giảm dần
    return sorted(match_map.items(), key=lambda x: len(x[1]), reverse=True)

def filter_question(question, model_name="gpt-4o-mini"):
    llm = ChatOpenAI(model_name=model_name, temperature=0)
    response = llm([
        SystemMessage(content="Xác định xem câu hỏi sau liên quan đến THUỐC hay BỆNH hoặc khác. Chỉ cần trả lời THUỐC hoặc BỆNH"),
        HumanMessage(content=question)
    ])
    return response.content.strip()

def get_label_of_entity(entity_id, driver):
    """
    Lấy danh sách nhãn (labels) của node có id tương ứng, bỏ qua '__ENTITY__'.
    Trả về danh sách labels (ví dụ: ['BỆNH'], ['CHỦ_ĐỀ'], ...).
    """

    with driver.session() as session:
        query = (
            "MATCH (n) "
            "WHERE n.id = $entity_id "
            "RETURN labels(n) AS labels "
            "LIMIT 1"
        )
        result = session.run(query, entity_id=entity_id)
        record = result.single()
        if record:
            labels = record["labels"]
            filtered_labels = [label for label in labels if label != "__Entity__"]
            return filtered_labels
    return []

def get_related_nodes(entity_id, driver):
    """
    Lấy danh sách các node có quan hệ outgoing với node có id là entity_id,
    tự động xác định label của node này và truy vấn theo label đó.
    """

    labels = get_label_of_entity(entity_id, driver)

    if not labels:
        return []
    label = labels[0]  # Chọn label đầu tiên (sau khi đã loại bỏ __Entity__)

    with driver.session() as session:
        query = (
            f"MATCH (n:`{label}` {{id: $entity_id}})-[r]->(m) "
            "RETURN DISTINCT m.id AS node_name"
        )
        result = session.run(query, entity_id=entity_id)
        related_nodes = [record["node_name"] for record in result]
        return related_nodes
    
def create_context_from_top_results(top_results, driver):
    def capitalize_sentences(text):
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return ' '.join(s.capitalize() for s in sentences)

    contexts = []

    for topic_name, related_nodes in top_results:
        def build_context(nodes):
            parts = []
            khainiem_text = None
            for attr in nodes:
                # Bỏ qua nếu node có label là BỆNH, THUỐC hoặc CHỦ_ĐỀ
                labels = get_label_of_entity(attr, driver)
                if not labels or any(lbl.upper() in {"BỆNH", "THUỐC", "CHỦ_ĐỀ"} for lbl in labels):
                    continue

                details = get_related_nodes(attr, driver)
                if details:
                    clean_details = [d.rstrip('.').strip().lower() for d in details]
                    attr_clean = attr.strip().lower()

                    if attr_clean.startswith("khái niệm"):
                        khainiem_text = ", ".join(sorted(set(clean_details)))
                    else:
                        # Ghép mỗi 2 chi tiết thành 1 câu
                        for i in range(0, len(clean_details), 2):
                            chunk = clean_details[i:i+2]
                            # Chuyển dấu chấm thành dấu phẩy trong từng detail trước khi merge
                            chunk = [d.replace('.', ',') for d in chunk]
                            merged = ", ".join(sorted(set(chunk)))
                            parts.append(f"{attr_clean} là {merged}")
            if khainiem_text:
                parts.insert(0, khainiem_text)
            context = ". ".join(parts) + "."
            return capitalize_sentences(context)

        # Tạo context chính
        main_context = build_context(related_nodes)

        # Đếm số câu trong main_context
        sentence_count = len([s for s in re.split(r'[.!?]\s*', main_context) if s.strip()])

        extra_sentence = ""
        if sentence_count < 2:
            all_nodes = get_related_nodes(topic_name, driver)
            extra_nodes = [n for n in all_nodes if n not in related_nodes]
            for attr in extra_nodes:
                labels = get_label_of_entity(attr, driver)
                if not labels or any(lbl.upper() in {"BỆNH", "THUỐC", "CHỦ_ĐỀ"} for lbl in labels):
                    continue

                details = get_related_nodes(attr, driver)
                if details:
                    clean_details = [d.strip().lower().replace('.', ',').rstrip(',').strip() for d in details]
                    merged_detail = ", ".join(sorted(set(clean_details)))
                    attr_clean = attr.strip().lower()
                    extra_sentence = capitalize_sentences(f"{attr_clean} là {merged_detail}.")
                    break

        # Gộp context đầy đủ
        full_context = main_context + " " + extra_sentence if extra_sentence else main_context
        contexts.append(full_context)

    return contexts


def get_top_topic_contexts(retriever, driver, question):
    results_dict, high_score_results = search_queries(
        retriever,
        sentence_to_keywords(question)
    )
    # tìm kiếm topics với node có điểm cao nhất
    max_score = max((item.metadata.get('score', 0) for item in high_score_results), default=0)
    highest_score_results = [item for item in high_score_results if item.metadata.get('score', 0) == max_score]
    topic_highest_score = find_entities_from_retriever_results(
        driver,
        highest_score_results
    )

    # tìm kiếm topics có count nhiều nhất
    highest_score_results = find_entities_from_retriever_results(
        driver,
        high_score_results
    )
    max_count = max(len(matches) for _, matches in highest_score_results)
    topic_highest_count = [
        (entity_id, matches)
        for entity_id, matches in highest_score_results
        if len(matches) == max_count
    ]

    # Gộp 2 danh sách kết quả theo entity_id
    merged_result = defaultdict(set)
    for entity_id, matches in topic_highest_score + topic_highest_count:
        merged_result[entity_id].update(matches)
    result = [(entity_id, list(matches)) for entity_id, matches in merged_result.items()]

    # Tạo danh sách context từ các kết quả tìm kiếm
    contexts = create_context_from_top_results(result, driver)

    documents = [
        Document(page_content=context, metadata={"source": idx})
        for idx, context in enumerate(contexts)
    ]
    # Tách văn bản thành các chunk nhỏ
    text_splitter_dot = RecursiveCharacterTextSplitter(
        chunk_size=412,
        chunk_overlap=0,
        separators=["."]
    )

    text_splitter_comma = RecursiveCharacterTextSplitter(
        chunk_size=412,
        chunk_overlap=0,
        separators=[","]
    )

    # Bước 1: Chia theo dấu chấm "."
    dot_chunks = text_splitter_dot.split_documents(documents)

    # Bước 2: Tiếp tục chia từng đoạn theo dấu phẩy ","
    final_chunks = []
    for chunk in dot_chunks:
        smaller_chunks = text_splitter_comma.split_documents([chunk])
        final_chunks.extend(smaller_chunks)

    # Trả về nội dung văn bản đã chia nhỏ
    return [doc.page_content for doc in final_chunks]    


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
        contexts = get_top_topic_contexts(retriever, driver, question)
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



