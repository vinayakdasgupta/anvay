# -*- coding: utf-8 -*-

import logging
import os
import re
from werkzeug.utils import secure_filename
from werkzeug.exceptions import HTTPException
import csv
import string
import nltk
import gensim
import numpy as np
import traceback
import gensim.corpora as corpora
from collections import Counter
from nltk.corpus import stopwords
from nltk.data import find

# Text processing utilities
def remove_punctuation(text):
    chars_to_remove = string.punctuation + '।॥“”‘’—–…•·―০১২৩৪৫৬৭৮৯'
    return re.sub(f"[{re.escape(chars_to_remove)}]", "", text)


# Define Bengali Unicode range and common punctuation
BENGALI_UNICODE_RANGE = r'\u0980-\u09FF'
ZERO_WIDTH = r'\u200B\u200C\u200D\u200E\u200F\u202A-\u202E\u2060\uFEFF'  # Zero-width characters to be removed
CUSTOM_PUNCTUATION = string.punctuation + '।॥“”‘’'

# Custom tokenizer that avoids breaking at zero-width characters
def custom_bengali_tokenize(text):
    # Remove unwanted punctuation and zero-width characters upfront
    text = re.sub(f"[{re.escape(CUSTOM_PUNCTUATION)}{ZERO_WIDTH}]", " ", text)
    
    # Retain only Bengali characters and spaces
    cleaned_text = ''.join(ch for ch in text if ch == ' ' or '\u0980' <= ch <= '\u09FF')
    
    # Tokenize the text by splitting on whitespace
    tokens = cleaned_text.split()
    
    # Optionally remove any malformed tokens (those containing zero-width characters)
    tokens = [t for t in tokens if not re.fullmatch(r'.*[\u200B\u200C\u200D\u200E\u200F\u202A-\u202E\u2060\uFEFF].*', t)]
    
    return tokens


def load_stopwords(file_path):
    if not os.path.exists(file_path): return set()
    with open(file_path, 'r', encoding='utf-8') as f:
        return set(line.strip() for line in f)

def remove_most_frequent_words(all_tokens, percent):
    flat = [t for doc in all_tokens for t in doc]
    freq = Counter(flat)
    n = int((percent / 100) * len(freq))
    remove_set = set(w for w, _ in freq.most_common(n))
    return [[t for t in doc if t not in remove_set] for doc in all_tokens]
def parse_hyperparam(val):
    if val.lower() == "auto":
        return None
    if val.lower() == "symmetric":
        return None
    try:
        return float(val)
    except:
        return None
    
def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
        return float(obj)
    elif isinstance(obj, np.int32) or isinstance(obj, np.int64):
        return int(obj)
    return obj    

def get_relevance_weighted_words(lda_model, corpus, dictionary, lambda_val=0.6, topn=10):
    word_freq = Counter()
    total_tokens = 0
    for doc in corpus:
        for word_id, freq in doc:
            word_freq[word_id] += freq
            total_tokens += freq
    p_w = {wid: freq / total_tokens for wid, freq in word_freq.items()}

    topic_terms = []
    for topic_id in range(lda_model.num_topics):
        topic = lda_model.get_topic_terms(topicid=topic_id, topn=None)
        relevance_scores = []
        for word_id, p_w_t in topic:
            p_w_global = p_w.get(word_id, 1e-12)
            relevance = lambda_val * p_w_t + (1 - lambda_val) * p_w_global
            relevance_scores.append((word_id, relevance))
        sorted_terms = sorted(relevance_scores, key=lambda x: -x[1])[:topn]
        term_words = [(dictionary[word_id], score) for word_id, score in sorted_terms]
        topic_terms.append((topic_id, term_words))

    return topic_terms

def generate_topic_labels(relevance_topics):
    labels = {}
    for topic_id, words in relevance_topics:
        top_terms = [word for word, _ in words[:2]]
        label = " / ".join(top_terms)
        labels[topic_id] = label
    return labels

def split_sentences_bengali(text):
    return [s.strip() for s in re.split(r"[।॥!?\.]", text) if s.strip()]

def initialize_roots_and_suffixes(folder):
    noun_roots = set(load_list_from_file(os.path.join(folder, 'noun_roots.txt')))
    verb_roots = set(load_list_from_file(os.path.join(folder, 'verb_roots.txt')))
    noun_suffixes = load_list_from_file(os.path.join(folder, 'noun_suffixes.txt'))
    verb_suffixes = load_list_from_file(os.path.join(folder, 'verb_suffixes.txt'))
    return noun_roots, verb_roots, noun_suffixes, verb_suffixes

# Load word files
def load_list_from_file(file_path):
    if not os.path.exists(file_path): return []
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def get_word_pos(word, verb_suffixes):
    return "VERB" if word.endswith(tuple(verb_suffixes)) else "NOUN"

def remove_suffix(word, suffix):
    return word[:-len(suffix)] if word.endswith(suffix) else word

def bengali_stemmer(word, noun_roots, verb_roots, noun_suffixes, verb_suffixes):
    wpos = get_word_pos(word, verb_suffixes)
    roots = noun_roots if wpos == "NOUN" else verb_roots
    suffixes = noun_suffixes if wpos == "NOUN" else verb_suffixes
    if word in roots: return word
    for suffix in suffixes:
        root = remove_suffix(word, suffix)
        if root in roots: return root
    return word

def stem_tokens(tokens_list, noun_roots, verb_roots, noun_suffixes, verb_suffixes):
    return [[bengali_stemmer(token, noun_roots, verb_roots, noun_suffixes, verb_suffixes) for token in tokens]
            for tokens in tokens_list]

#TESTS

def get_sliding_windows(tokens, window_size=300, stride=100):
    """
    Segment a list of tokens into overlapping windows.
    Each window is a list of tokens.
    """
    return [tokens[i:i+window_size] for i in range(0, len(tokens) - window_size + 1, stride)]

