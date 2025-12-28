import re
from typing import Optional


def preprocess_for_fasttext(text: str) -> str:
    """
    Official FastText preprocessing adapted for Steam reviews.
    """
    if not isinstance(text, str):
        return ''
    
    # Steam-specific: Remove URLs first
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    
    # Lowercase
    text = text.lower()
    
    # Normalize smart quotes to regular apostrophe
    text = re.sub(r"['′''`]", "'", text)
    
    # Add space around apostrophes
    text = re.sub(r"'", " ' ", text)
    
    # Remove double quotes including smart quotes
    text = re.sub(r'["""]', '', text)
    
    # Add space around periods
    text = re.sub(r'\.', ' . ', text)
    
    # Remove <br /> tags
    text = re.sub(r'<br\s*/?>', ' ', text, flags=re.IGNORECASE)
    
    # Add space around commas
    text = re.sub(r',', ' , ', text)
    
    # Add space around parentheses
    text = re.sub(r'\(', ' ( ', text)
    text = re.sub(r'\)', ' ) ', text)
    
    # Add space around exclamation marks
    text = re.sub(r'!', ' ! ', text)
    
    # Add space around question marks
    text = re.sub(r'\?', ' ? ', text)
    
    # Remove semicolons and replace with space
    text = re.sub(r';', ' ', text)
    
    # Remove colons and replace with space
    text = re.sub(r':', ' ', text)
    
    # Remove any remaining HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def preprocess_for_fasttext_classification(text: str, label: Optional[str] = None) -> str:
    """
    Preprocessing for FastText supervised classification.
    """
    processed = preprocess_for_fasttext(text)
    
    if label is not None:
        return f"__label__{label} {processed}"
    
    return processed
