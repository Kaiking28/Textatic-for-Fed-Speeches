"""
textastic.py: An extensible framework for comparative text analysis
Federal Reserve Speech Analysis Framework
"""

from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import re
import squarify
from textblob import TextBlob


class Textastic:

    def __init__(self):
        self.data = defaultdict(dict)
        self.stop_words = set()

    def load_stop_words(self, stopfile=None, stop_list=None):
        if stopfile:
            with open(stopfile, 'r', encoding='utf-8') as f:
                self.stop_words = set(word.strip().lower() for word in f)
        elif stop_list:
            self.stop_words = set(word.lower() for word in stop_list)
        print(f"Loaded {len(self.stop_words)} stop words")

    @staticmethod
    def default_parser(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            text = f.read()

        blob = TextBlob(text)
        sentiment = blob.sentiment.polarity

        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()

        words = text.split()
        wordcount = Counter(words)
        numwords = len(words)
        unique_words = len(wordcount)

        results = {
            'wordcount': wordcount,
            'numwords': numwords,
            'unique_words': unique_words,
            'text': text,
            'sentiment': sentiment
        }

        return results

    def load_text(self, filename, label=None, parser=None):
        if parser is None:
            results = Textastic.default_parser(filename)
        else:
            results = parser(filename)

        if label is None:
            label = filename.split('/')[-1].replace('.txt', '')

        if self.stop_words and 'wordcount' in results:
            filtered_wordcount = Counter()
            for word, count in results['wordcount'].items():
                if word not in self.stop_words and len(word) > 0:
                    filtered_wordcount[word] = count
            results['wordcount'] = filtered_wordcount

        for k, v in results.items():
            self.data[k][label] = v

        print(f"Loaded '{label}': {results.get('numwords', 0)} words")

    # Sankey Diagram Visualization
    def wordcount_sankey(self, word_list=None, k=5):
        print("Order of labels:", list(self.data['wordcount'].keys()))

        text_labels = list(self.data['wordcount'].keys())

        if word_list:
            words = list(word_list)
        else:
            words = []
            seen = set()
            for label in text_labels:
                wordcount = self.data['wordcount'][label]
                top_words = [word for word, _ in wordcount.most_common(k)]
                for word in top_words:
                    if word not in seen:
                        words.append(word)
                        seen.add(word)

        sources = []
        targets = []
        values = []

        node_labels = text_labels + words

        for text_idx, label in enumerate(text_labels):
            wordcount = self.data['wordcount'][label]
            for word in words:
                count = wordcount.get(word, 0)
                if count > 0:
                    sources.append(text_idx)
                    targets.append(len(text_labels) + words.index(word))
                    values.append(count)

        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                label=node_labels,
                color=['lightblue'] * len(text_labels) + ['lightcoral'] * len(words)
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values
            )
        )])
        fig.update_layout(
            title=f"Text-to-Word Sankey Diagram (Chronological Order)",
            font_size=11,
            height=600
        )
        fig.show()

    # Treemap Subplots Visualization
    def word_frequency_subplots(self, top_n=10):
        if 'wordcount' not in self.data:
            print("No word count data available!")
            return

        try:
            import squarify
        except ImportError:
            print("Error: squarify not installed. Run: pip install squarify")
            return

        wordcounts = self.data['wordcount']
        n_texts = len(wordcounts)

        cols = min(3, n_texts)
        rows = (n_texts + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))

        if n_texts == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten() if hasattr(axes, 'flatten') else axes
        else:
            axes = axes.flatten()

        for idx, (label, wordcount) in enumerate(wordcounts.items()):
            ax = axes[idx]

            top_words = wordcount.most_common(top_n)
            if top_words:
                words, counts = zip(*top_words)
            else:
                words, counts = [], []

            if len(words) > 0:
                colors = plt.cm.viridis([i / len(words) for i in range(len(words))])

                squarify.plot(
                    sizes=counts,
                    label=words,
                    alpha=0.8,
                    ax=ax,
                    color=colors,
                    text_kwargs={'fontsize': 9, 'weight': 'bold'}
                )

            ax.set_title(f'{label}\n({self.data["numwords"][label]} total words)',
                         fontsize=11, fontweight='bold', pad=10)
            ax.axis('off')

        for idx in range(n_texts, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        plt.show()

    # Comparative Bar Chart Visualization
    def comparative_word_overlay(self, words):
        if 'wordcount' not in self.data:
            print("No word count data available!")
            return

        labels = list(self.data['wordcount'].keys())
        x = range(len(words))
        width = 0.8 / len(labels)

        fig, ax = plt.subplots(figsize=(12, 6))

        for idx, label in enumerate(labels):
            wordcount = self.data['wordcount'][label]
            counts = [wordcount.get(word, 0) for word in words]
            offset = width * idx - (0.8 / 2) + width / 2
            ax.bar([i + offset for i in x], counts, width, label=label, alpha=0.8)

        ax.set_xlabel('Words')
        ax.set_ylabel('Frequency')
        ax.set_title('Comparative Word Frequencies Across Texts')
        ax.set_xticks(x)
        ax.set_xticklabels(words, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.show()

    # Trend Line Visualization
    def word_trends_over_time(self, words, years):
        if 'wordcount' not in self.data:
            print("No word count data available!")
            return

        labels = list(self.data['wordcount'].keys())

        fig, ax = plt.subplots(figsize=(14, 7))

        colors = plt.cm.Set2(range(len(words)))
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']

        for idx, word in enumerate(words):
            counts = []
            for label in labels:
                wordcount = self.data['wordcount'][label]
                counts.append(wordcount.get(word, 0))

            ax.plot(years, counts, marker=markers[idx % len(markers)],
                    linewidth=2.5, label=word.capitalize(),
                    color=colors[idx], markersize=8)

        ax.set_xlabel('Year', fontsize=12, fontweight='bold')
        ax.set_ylabel('Word Frequency', fontsize=12, fontweight='bold')
        ax.set_title('Key Economic Terms: Frequency Over Time',
                     fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='best', fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    # Sentiment Analysis Visualization
    def sentiment_analysis_bar(self):
        if 'sentiment' not in self.data:
            print("No sentiment data available!")
            return

        labels = list(self.data['sentiment'].keys())
        sentiments = [self.data['sentiment'][label] for label in labels]

        for label, sentiment in zip(labels, sentiments):
            print(f"{label}: Sentiment = {sentiment:.3f}")

        fig, ax = plt.subplots(figsize=(14, 7))

        colors = ['red' if s < 0 else 'lightgreen' if s < 0.05 else 'green'
                  for s in sentiments]

        bars = ax.bar(range(len(labels)), sentiments, color=colors,
                      alpha=0.7, edgecolor='black', linewidth=1.5)

        for idx, (bar, sentiment) in enumerate(zip(bars, sentiments)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{sentiment:.3f}',
                    ha='center', va='bottom' if height > 0 else 'top',
                    fontsize=10, fontweight='bold')

        ax.set_xlabel('Federal Reserve Speech', fontsize=12, fontweight='bold')
        ax.set_ylabel('Sentiment Polarity', fontsize=12, fontweight='bold')
        ax.set_title('Sentiment Analysis of Federal Reserve Speeches Over Time',
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(min(sentiments) - 0.05, max(sentiments) + 0.05)

        plt.tight_layout()
        plt.show()