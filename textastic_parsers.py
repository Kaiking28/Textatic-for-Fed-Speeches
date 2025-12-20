
"""
textastic_parsers.py
Custom domain-specific parsers for different text formats.
"""

import json
import re
from collections import Counter
from textblob import TextBlob


def json_parser(filename):
    """
    Example parser for JSON files containing text data.
    
    Args:
        filename: Path to JSON file
        
    Returns:
        Dictionary with wordcount and numwords
    """
    with open(filename, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    
    text = raw.get('text', '')
    words = text.split(" ")
    wc = Counter(words)
    num = len(words)
    
    return {
        'wordcount': wc, 
        'numwords': num,
        'text': text
    }


def fed_speech_parser(filename):
    """
    Custom parser for Federal Reserve speeches.
    Handles Fed-specific formatting and extracts relevant content.

    Args:
        filename: Path to the Fed speech text file

    Returns:
        dict: Processed data including wordcount, stats, and sentiment
    """
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()

    # Store original for sentiment analysis
    original_text = text

    # Clean text: lowercase, remove punctuation, normalize whitespace
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenize
    words = text.split()

    # Count words
    wordcount = Counter(words)
    numwords = len(words)
    unique_words = len(wordcount)

    # Calculate sentiment on original text
    blob = TextBlob(original_text)
    sentiment = blob.sentiment.polarity

    results = {
        'wordcount': wordcount,
        'numwords': numwords,
        'unique_words': unique_words,
        'text': text,
        'sentiment': sentiment
    }

    return results

