# -*- coding: utf-8 -*-
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.data import find
import string
from collections import Counter

# Download required NLTK resources
def download_nltk_resources():
    try:
        find('corpora/stopwords')
        find('tokenizers/punkt')
    except LookupError:
        nltk.download('stopwords')
        nltk.download('punkt')

download_nltk_resources()

# Function to load lists from text files
def load_list_from_file(file_path):
    """
    Reads a list of items from a text file, one item per line.
    """
    if not os.path.exists(file_path):
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines() if line.strip()]

# Create necessary directories
def ensure_directories_exist(*folders):
    for folder in folders:
        os.makedirs(folder, exist_ok=True)

UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
STOPWORDS_FOLDER = 'stopwords'

ensure_directories_exist(UPLOAD_FOLDER, RESULT_FOLDER, STOPWORDS_FOLDER)

# Stopwords
stop_words = set(stopwords.words('bengali'))

# Load noun roots, verb roots, noun suffixes, and verb suffixes
NOUN_ROOTS_FILE = os.path.join(STOPWORDS_FOLDER, 'noun_roots.txt')
VERB_ROOTS_FILE = os.path.join(STOPWORDS_FOLDER, 'verb_roots.txt')
NOUN_SUFFIXES_FILE = os.path.join(STOPWORDS_FOLDER, 'noun_suffixes.txt')
VERB_SUFFIXES_FILE = os.path.join(STOPWORDS_FOLDER, 'verb_suffixes.txt')

NOUN_ROOTS = set(load_list_from_file(NOUN_ROOTS_FILE))
VERB_ROOTS = set(load_list_from_file(VERB_ROOTS_FILE))
NOUN_SUFFIXES = load_list_from_file(NOUN_SUFFIXES_FILE)
VERB_SUFFIXES = load_list_from_file(VERB_SUFFIXES_FILE)

def filter_bengali(text):
    # Keep only Bengali characters and whitespace, remove English characters and punctuation
    return re.sub(r'[^অ-হু_ং ঃ০-৯\s]+', '', text)

# Helper functions for stemming
def is_valid_root(word, root_dict):
    return word in root_dict

def get_word_pos(word):
    """
    Determine the part of speech (POS) of a word based on suffixes.
    """
    if word.endswith(tuple(NOUN_SUFFIXES)):
        return "NOUN"
    elif word.endswith(tuple(VERB_SUFFIXES)):
        return "VERB"
    else:
        return "NOUN"  # Default to NOUN

def remove_suffix(word, suffix):
    """
    Remove a suffix from a word if present.
    """
    if word.endswith(suffix):
        return word[:-len(suffix)]
    return word

def bengali_stemmer(word):
    """
    Rule-based stemming for Bengali words.
    """
    WPOS = get_word_pos(word)

    if WPOS == "NOUN":
        if is_valid_root(word, NOUN_ROOTS):
            return word
        for suffix in NOUN_SUFFIXES:
            possible_root = remove_suffix(word, suffix)
            if is_valid_root(possible_root, NOUN_ROOTS):
                return possible_root

    elif WPOS == "VERB":
        if is_valid_root(word, VERB_ROOTS):
            return word
        for suffix in VERB_SUFFIXES:
            possible_root = remove_suffix(word, suffix)
            if is_valid_root(possible_root, VERB_ROOTS):
                return possible_root

    return word

def remove_punctuation(text):
    """
    Removes punctuation from text, including Bengali-specific punctuation.
    """
    bengali_punctuations = "।॥“”‘’—…‹›«»!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    all_punctuations = string.punctuation + bengali_punctuations
    return re.sub(f"[{re.escape(all_punctuations)}]", "", text)

# Function to remove most frequent words based on percentage
def remove_most_frequent_words(all_tokens, percent_most_common):
    """
    Removes the most frequent words from the tokenized texts.
    """
    flat_tokens = [token for tokens in all_tokens for token in tokens]
    word_freq = Counter(flat_tokens)
    num_to_remove = int((percent_most_common / 100) * len(word_freq))
    most_common_words = set(word for word, _ in word_freq.most_common(num_to_remove))
    return [[token for token in tokens if token not in most_common_words] for tokens in all_tokens]

# Function to preprocess text
def preprocess_text(text, stop_words=None, use_stemming=False):
    """
    Preprocesses a single text by removing punctuation, tokenizing, removing stopwords, and stemming.
    """
    if not text.strip():
        return []
    
    text = filter_bengali(text)
    text = remove_punctuation(text)
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if len(token) > 1]

    if stop_words:
        tokens = [word for word in tokens if word not in stop_words]

    if use_stemming:
        tokens = [bengali_stemmer(token) for token in tokens]

    return tokens

# Function to preprocess multiple files
def load_and_preprocess_files(file_paths, stop_words=None, use_stemming=False):
    """
    Reads and preprocesses all text files.
    """
    all_tokens = []
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            if not text.strip():
                print(f"Warning: File '{file_path}' is empty or contains only whitespace.")
                all_tokens.append([])
                continue
            tokens = preprocess_text(text, stop_words, use_stemming)
            all_tokens.append(tokens)
    return all_tokens

# Function to remove most frequent words
def filter_frequent_words(all_tokens, percent_most_common):
    return remove_most_frequent_words(all_tokens, percent_most_common)
