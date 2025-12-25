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
    
    # Lowercase (tr '[:upper:]' '[:lower:]')
    text = text.lower()
    
    # Normalize smart quotes to regular apostrophe (from get-wikimedia.sh)
    # s/'/'/g -e s/′/'/g
    text = re.sub(r"['′''`]", "'", text)
    
    # Add space around apostrophes (s/'/ ' /g)
    text = re.sub(r"'", " ' ", text)
    
    # Remove double quotes including smart quotes (s/"//g)
    # From get-wikimedia.sh: s/"/\"/g -e s/"/\"/g then s/"/ " /g
    text = re.sub(r'["""]', '', text)
    
    # Add space around periods (s/\./ \. /g)
    text = re.sub(r'\.', ' . ', text)
    
    # Remove <br /> tags (s/<br \/>/ /g)
    text = re.sub(r'<br\s*/?>', ' ', text, flags=re.IGNORECASE)
    
    # Add space around commas (s/,/ , /g)
    text = re.sub(r',', ' , ', text)
    
    # Add space around parentheses (s/(/ ( /g and s/)/ ) /g)
    text = re.sub(r'\(', ' ( ', text)
    text = re.sub(r'\)', ' ) ', text)
    
    # Add space around exclamation marks (s/\!/ \! /g)
    text = re.sub(r'!', ' ! ', text)
    
    # Add space around question marks (s/\?/ \? /g)
    text = re.sub(r'\?', ' ? ', text)
    
    # Remove semicolons and replace with space (s/\;/ /g)
    text = re.sub(r';', ' ', text)
    
    # Remove colons and replace with space (s/\:/ /g)
    text = re.sub(r':', ' ', text)
    
    # Steam-specific: Remove any remaining HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Collapse multiple spaces (tr -s " ")
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
