from nltk.tokenize import sent_tokenize
import pandas as pd
import re
import string
from pyvi import ViTokenizer
import nltk

# Download necessary NLTK resources
nltk.download('punkt')


def clean_text(text, keep_period=False):
    """Clean text by tokenizing, lowercasing, removing punctuation, and emojis."""
    text = ViTokenizer.tokenize(text).lower()
    text = remove_punctuation(text, keep_period)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_punctuation(text, keep_period=False):
    """Remove punctuation, emojis, and optionally keep periods."""
    if keep_period:
        punctuation_to_remove = string.punctuation.replace(".", "").replace("_", "")
    else:
        punctuation_to_remove = string.punctuation.replace("_", "")

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
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642"
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub('', text)
    
    return text

def transform_load(df):
    """Transform raw data to usable text, apply processing functions, and handle splitting."""
    df = df.fillna('')
    new_rows = []
    
    for _, row in df.iterrows():
        title = row['Question']
        content = row['Answer']
        
        # Clean the text data and create new row
        new_row = {
            'Question': clean_text(title, keep_period=True),  # Cột Question
            'Context': clean_text(content),  # Cột Answer
        }
        new_rows.append(new_row)
    
    new_df = pd.DataFrame(new_rows)
    return new_df

def save_to_csv(processed_news, output_path):
    """Save cleaned data to CSV."""
    processed_news.to_csv(output_path, index=False, encoding='utf-8')

# Example usage
if __name__ == "__main__":
    input_csv_path = '../csv/processed_tamanh_hospital_cleaned.csv'
    output_csv_path = '../csv/processed_tamanh_hospital_cleaned_full.csv'
    
    df = pd.read_csv(input_csv_path)
    processed_news = transform_load(df)
    save_to_csv(processed_news, output_csv_path)

    print(f"Processed data has been saved to {output_csv_path}.")
