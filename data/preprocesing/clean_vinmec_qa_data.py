from nltk.tokenize import sent_tokenize
import pandas as pd
import re
import string
from underthesea import word_tokenize 
import nltk
import os

nltk.download('punkt', quiet=True)

def clean_text(text, keep_period=True):
    """
    Clean text by tokenizing Vietnamese text, lowercasing, removing punctuation, and emojis.
    Using underthesea for better compound word handling.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    
    tokenized = word_tokenize(text)
    tokenized_text = " ".join([t.replace(" ", "_") for t in tokenized])
    tokenized_text = tokenized_text.lower()
    
    cleaned_text = remove_punctuation(tokenized_text, keep_period)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    return cleaned_text

def remove_punctuation(text, keep_period=True):
    """Remove punctuation, emojis, and optionally keep periods."""
    if keep_period:
        punctuation_to_remove = string.punctuation.replace(".", "").replace("_", "")
    else:
        punctuation_to_remove = string.punctuation.replace("_", "")  # Keep underscores for compound words

    translator = str.maketrans('', '', punctuation_to_remove)
    text = text.translate(translator)

    emoji_pattern = re.compile(
        "[" 
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001F926-\U0001F937"
        u"\U00010000-\U0010FFFF"
        u"\u2640-\u2642"
        u"\u2600-\u2B55"
        u"\u200D"
        u"\u23CF"
        u"\u23E9"
        u"\u231A"
        u"\uFE0F"  # dingbats
        u"\u3030"
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub('', text)
    
    return text

def process_qa_data(input_file, output_file):
    """
    Process the MSD QA dataset by cleaning all three columns:
    question, context, and answer.
    """
    print(f"Processing file: {input_file}")
    
    try:
        df = pd.read_csv(input_file)
        original_count = len(df)
        print(f"Loaded {original_count} QA pairs from {input_file}")
    except Exception as e:
        print(f"Error loading file: {e}")
        return
    
    required_columns = ['question', 'context', 'answer']
    if not all(column in df.columns for column in required_columns):
        print(f"File must contain these columns: {required_columns}")
        return
    
    df = df.fillna('')
    
    print("Cleaning question column...")
    df['question'] = df['question'].apply(lambda x: clean_text(x, keep_period=True))
    
    print("Cleaning context column...")
    df['context'] = df['context'].apply(lambda x: clean_text(x, keep_period=True))
    
    print("Cleaning answer column...")
    df['answer'] = df['answer'].apply(lambda x: clean_text(x, keep_period=True))
    
    df = df.replace('', pd.NA)
    df = df.dropna(subset=required_columns)
    
    processed_count = len(df)
    print(f"Processed {processed_count} QA pairs (removed {original_count - processed_count} empty entries)")
    
    try:
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"Processed data saved to {output_file}")
    except Exception as e:
        print(f"Error saving processed data: {e}")

def load_stopwords(stopwords_file):
    """Load Vietnamese stopwords from a file."""
    try:
        with open(stopwords_file, 'r', encoding='utf-8') as f:
            stopwords = set(line.strip() for line in f if line.strip())
        return stopwords
    except Exception as e:
        print(f"Error loading stopwords: {e}")
        return set()

def remove_stopwords(text, stopwords):
    """Remove stopwords from text."""
    if not stopwords:
        return text
    words = text.split()
    filtered_words = [word for word in words if word not in stopwords]
    return ' '.join(filtered_words)

def process_with_stopwords_removal(input_file, output_file, stopwords_file):
    """Process QA data with additional stopwords removal."""
    stopwords = load_stopwords(stopwords_file)
    if not stopwords:
        print("No stopwords loaded, proceeding with standard cleaning only.")
    else:
        print(f"Loaded {len(stopwords)} stopwords")
    
    temp_output = input_file.replace('.csv', '_temp.csv')
    process_qa_data(input_file, temp_output)
    
    if stopwords:
        try:
            df = pd.read_csv(temp_output)
            
            print("Removing stopwords from question column...")
            df['question'] = df['question'].apply(lambda x: remove_stopwords(x, stopwords) if isinstance(x, str) else x)
            
            print("Removing stopwords from context column...")
            df['context'] = df['context'].apply(lambda x: remove_stopwords(x, stopwords) if isinstance(x, str) else x)
            
            print("Removing stopwords from answer column...")
            df['answer'] = df['answer'].apply(lambda x: remove_stopwords(x, stopwords) if isinstance(x, str) else x)
            
            # Save final processed data to CSV
            df.to_csv(output_file, index=False, encoding='utf-8')
            print(f"Processed data with stopwords removal saved to {output_file}")
            
            # Remove temporary file
            if os.path.exists(temp_output):
                os.remove(temp_output)
                
        except Exception as e:
            print(f"Error in stopwords removal phase: {e}")
    else:
        # If no stopwords to remove, just rename the temp file
        if os.path.exists(temp_output):
            try:
                os.rename(temp_output, output_file)
            except Exception as e:
                print(f"Error renaming temp file: {e}")

if __name__ == "__main__":
    input_csv_path = 'E:/university/TLCN/ChatBot/data/csv/qa_disease.csv'
    output_csv_path = 'E:/university/TLCN/ChatBot/data/csv/processed_qa_disease.csv'
    output_with_stopwords_removed = 'E:/university/TLCN/ChatBot/data/csv/processed_msd_qa_data_clean.csv'
    stopwords_file = 'E:/university/TLCN/ChatBot/data/vietnamese-stopwords.txt'
    
    process_qa_data(input_csv_path, output_csv_path)
    
    user_choice = input("Do you want to proceed with stopwords removal? (y/n): ")
    if user_choice.lower() == 'y':
        process_with_stopwords_removal(input_csv_path, output_with_stopwords_removed, stopwords_file)
    else:
        print("Skipping stopwords removal. Basic processing completed.")