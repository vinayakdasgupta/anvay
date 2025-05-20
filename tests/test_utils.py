# tests/test_utils.py

import os
import tempfile
import pytest
import numpy as np
from collections import Counter
from gensim.corpora import Dictionary

from utils import (
    remove_punctuation,
    custom_bengali_tokenize,
    stem_tokens,
    bengali_stemmer,
    parse_hyperparam,
    load_stopwords,
    get_relevance_weighted_words,
    generate_topic_labels
)

# -------- remove_punctuation --------
def test_remove_punctuation_basic():
    assert remove_punctuation("আমি, তুমি। সে!") == "আমি তুমি সে"

def test_remove_punctuation_only():
    assert remove_punctuation("!?॥।।…") == ""

def test_remove_punctuation_empty():
    assert remove_punctuation("") == ""

# -------- custom_bengali_tokenize --------
def test_tokenizer_simple():
    text = "সে বলল: আমি যাব না।"
    tokens = custom_bengali_tokenize(text)
    assert "আমি" in tokens
    assert "যাব" in tokens

def test_tokenizer_empty():
    assert custom_bengali_tokenize("") == []

def test_tokenizer_only_punct():
    assert custom_bengali_tokenize("!@#$%") == []

def test_tokenizer_zero_width():
    zero_width = "ন\u200cম\u200dস্কার"
    tokens = custom_bengali_tokenize(zero_width)
    assert "নমস্কার" not in tokens  # removed due to malformed composition

# -------- bengali_stemmer & stem_tokens --------
def test_bengali_stemmer_known():
    assert bengali_stemmer("যাচ্ছি") != ""  # mapped in lemma dict

def test_bengali_stemmer_unknown():
    word = "অজানা_শব্দ"
    assert bengali_stemmer(word) == word

def test_stem_tokens_list():
    tokens = [["যাচ্ছি", "পড়ছি"]]
    result = stem_tokens(tokens, None, None, None, None)
    assert isinstance(result, list)
    assert len(result[0]) == 2

# -------- parse_hyperparam --------
def test_parse_hyperparam_symmetric():
    assert parse_hyperparam("symmetric") is None

def test_parse_hyperparam_auto():
    assert parse_hyperparam("auto") is None

def test_parse_hyperparam_float():
    assert parse_hyperparam("0.01") == 0.01

def test_parse_hyperparam_invalid():
    assert parse_hyperparam("invalid") is None

# -------- load_stopwords --------
def test_load_stopwords_valid():
    with tempfile.NamedTemporaryFile(mode='w+', encoding='utf-8', delete=False) as f:
        f.write("আমি\nতুমি\n")
        f.flush()
        path = f.name
    result = load_stopwords(path)
    os.remove(path)
    assert "আমি" in result

def test_load_stopwords_missing():
    result = load_stopwords("non_existent_file.txt")
    assert result == set()

# -------- get_relevance_weighted_words --------
def test_get_relevance_weighted_words_minimal():
    # Build dummy LDA model with gensim
    from gensim.models import LdaModel

    texts = [["আমি", "তুমি"], ["সে", "আমি"]]
    dictionary = Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=2, passes=5)

    topics = get_relevance_weighted_words(lda, corpus, dictionary, lambda_val=0.6, topn=5)
    assert isinstance(topics, list)
    assert len(topics) == 2
    assert all(isinstance(t[1], list) for t in topics)

# -------- generate_topic_labels --------
def test_generate_topic_labels_simple():
    sample_topics = [
        (0, [("আম", 0.5), ("গাছ", 0.3)]),
        (1, [("নদী", 0.6), ("পাহাড়", 0.4)])
    ]
    labels = generate_topic_labels(sample_topics)
    assert labels[0] == "আম / গাছ"
    assert labels[1] == "নদী / পাহাড়"
