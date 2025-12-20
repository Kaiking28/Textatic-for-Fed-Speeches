"""
textastic_app.py - Updated with Sentiment Analysis
Application for analyzing Federal Reserve speeches using the Textastic framework.

Instructions:
1. Place your Fed speech .txt files in the same directory as this script
2. Update the speeches list below with your actual filenames
3. Run: python textastic_app.py
"""

from textastic import Textastic
import textastic_parsers as tp
import pprint as pp

STOP_WORDS = [
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her",
    "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs",
    "themselves", "what", "which", "who", "whom", "this", "that", "these", "those",
    "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if",
    "or", "because", "as", "until", "while", "of", "at", "by", "for", "with",
    "about", "against", "between", "into", "through", "during", "before", "after",
    "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over",
    "under", "again", "further", "then", "once", "here", "there", "when", "where",
    "why", "how", "all", "any", "both", "each", "few", "more", "most", "other",
    "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
    "very", "s", "t", "can", "will", "just", "don", "should", "now", "would", "could",
    "may", "might", "must", "shall", "ought", "also", "make", "made", "well", "one",
    "two", "said"
]


def main():
    print("\n" + "=" * 70)
    print(" TEXTASTIC - Federal Reserve Speech Analysis Framework")
    print("=" * 70 + "\n")

    tt = Textastic()

    print("Step 1: Loading stop words...")
    tt.load_stop_words(stop_list=STOP_WORDS)
    print()

    print("Step 2: Loading Federal Reserve speeches...")
    print("-" * 70)

    speeches = [
        ('yellen_2016.txt', 'Yellen 2016'),
        ('yellen_2017.txt', 'Yellen 2017'),
        ('powell_2018.txt', 'Powell 2018'),
        ('powell_2019.txt', 'Powell 2019'),
        ('powell_2020.txt', 'Powell 2020'),
        ('powell_2022.txt', 'Powell 2022'),
        ('powell_2023.txt', 'Powell 2023'),
        ('powell_2024.txt', 'Powell 2024'),
        ('powell_2025.txt', 'Powell 2025')
    ]

    for filename, label in speeches:
        try:
            tt.load_text(filename, label=label, parser=tp.fed_speech_parser)
        except FileNotFoundError:
            print(f"Warning: '{filename}' not found - skipping")

    print("\n" + "-" * 70)
    print("Step 3: Analyzing data...")
    print()

    print("Data structure overview:")
    print("Keys in data:", list(tt.data.keys()))
    print("Documents loaded:", len(tt.data.get('wordcount', {})))
    print()

    print("\n" + "=" * 70)
    print(" VISUALIZATIONS")
    print("=" * 70 + "\n")

    # Sankey Diagram
    print("Generating Visualization 1: Text-to-Word Sankey Diagram...")
    key_words = ['inflation', 'labor', 'growth', 'financial']
    tt.wordcount_sankey(word_list=key_words)

    # Treemap Subplots
    print("\nGenerating Visualization 2: Word Frequency Subplots...")
    tt.word_frequency_subplots(top_n=5)

    # Comparative Bar Chart
    print("\nGenerating Visualization 3: Comparative Word Frequencies...")
    economic_terms = [
        'inflation', 'economy', 'policy', 'rate', 'growth',
        'market', 'financial', 'bank', 'labor', 'risk'
    ]
    tt.comparative_word_overlay(economic_terms)

    years = [2016, 2017, 2018, 2019, 2020, 2022, 2023, 2024, 2025]

    # Trend Line Analysis
    print("\nGenerating Visualization 4: Key Economic Terms Over Time...")
    economic_keywords = ['inflation', 'labor', 'financial', 'growth']
    tt.word_trends_over_time(economic_keywords, years)

    # Sentiment Analysis
    print("\nGenerating Visualization 5: Sentiment Analysis Bar Chart...")
    tt.sentiment_analysis_bar()


if __name__ == "__main__":
    main()