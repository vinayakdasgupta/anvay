# -*- coding: utf-8 -*-

import os
import json
import logging
import re
import string
import gensim

from utils import remove_most_frequent_words


# ------------------------------------------------------------------
# Tamil Unicode + punctuation setup (mirrors Bengali / Hindi logic)
# ------------------------------------------------------------------

TAMIL_UNICODE_RANGE = r'\u0B80-\u0BFF'
ZERO_WIDTH = r'\u200B\u200C\u200D\u200E\u200F\u202A-\u202E\u2060\uFEFF'
CUSTOM_PUNCTUATION = string.punctuation + '।॥“”‘’—–…•·―'


def custom_tamil_tokenize(text):
    """
    Tokeniser modelled exactly on custom_bengali_tokenize
    and custom_hindi_tokenize, adapted for Tamil.
    """

    # Remove punctuation and zero-width chars
    text = re.sub(
        f"[{re.escape(CUSTOM_PUNCTUATION)}{ZERO_WIDTH}]",
        " ",
        text
    )

    # Retain only Tamil characters and spaces
    cleaned = "".join(
        ch for ch in text
        if ch == " " or "\u0B80" <= ch <= "\u0BFF"
    )

    tokens = cleaned.split()

    # Final sanity filter (same paranoia as Bengali / Hindi)
    tokens = [
        t for t in tokens
        if not re.fullmatch(
            r".*[\u200B\u200C\u200D\u200E\u200F\u202A-\u202E\u2060\uFEFF].*",
            t
        )
    ]

    return tokens


# ------------------------------------------------------------------
# Stopwords
# ------------------------------------------------------------------

def _get_tamil_stopwords():
    """
    Tamil stopwords are handled via custom lists only.
    No NLTK stopwords are used.
    """
    return set()


# ------------------------------------------------------------------
# Stem dictionary loader
# ------------------------------------------------------------------

def _load_tamil_stem_dict():
    """
    Load Tamil stemming dictionary (word -> stem).
    """
    base_dir = os.path.dirname(os.path.dirname(__file__))  # anvay/
    dict_path = os.path.join(
        base_dir,
        "data",
        "tamil_lemma_dict.json"
    )

    with open(dict_path, "r", encoding="utf-8") as f:
        return json.load(f)



TAMIL_STEM_DICT = _load_tamil_stem_dict()


def tamil_stem(token):
    """
    Dictionary-based Tamil stemming.
    Falls back to surface form if not found.
    """
    return TAMIL_STEM_DICT.get(token, token)


# ------------------------------------------------------------------
# Main preprocessing function
# ------------------------------------------------------------------

def preprocess_ta(
    file_paths,
    remove_stopwords,
    custom_stopwords,
    normaliser,          # Normaliser or None
    percent,
    ngram
):
    """
    Tamil preprocessing.

    Tokenisation:
    - Custom Tamil tokenizer (Bengali-style logic)

    Normalisation:
    - ta_stem via dictionary lookup

    Order:
    - filter → stem (fixed)
    """

    # ------------------------------------------------------------------
    # Validate normaliser
    # ------------------------------------------------------------------

    if normaliser is not None and normaliser.key != "ta_stem":
        raise ValueError(
            f"Unsupported normaliser for Tamil: {normaliser.key}"
        )

    use_stemming = normaliser is not None

    # ------------------------------------------------------------------
    # Build stopword set
    # ------------------------------------------------------------------

    stop_words = set()

    if remove_stopwords:
        stop_words.update(_get_tamil_stopwords())

    if custom_stopwords:
        stop_words.update(custom_stopwords)

    logging.info(
        "Tamil tokenisation: custom Tamil tokenizer"
    )

    if use_stemming:
        logging.info(
            "Lexical normalisation: ta_stem (dictionary-based, fixed order)"
        )
    else:
        logging.info("Lexical normalisation: None")

    # ------------------------------------------------------------------
    # Process documents
    # ------------------------------------------------------------------

    all_tokens = []
    raw_texts = []
    doc_names = []

    for path in file_paths:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read().lower()

        tokens = custom_tamil_tokenize(raw)

        # Filter first
        filtered = [
            t for t in tokens
            if len(t) > 1 and t not in stop_words
        ]

        # Optional stemming
        if use_stemming:
            tokens = [tamil_stem(t) for t in filtered]
        else:
            tokens = filtered

        all_tokens.append(tokens)
        raw_texts.append(raw)
        doc_names.append(os.path.basename(path))

    # ------------------------------------------------------------------
    # Corpus-level frequency pruning
    # ------------------------------------------------------------------

    if percent > 0:
        all_tokens = remove_most_frequent_words(all_tokens, percent)

    # ------------------------------------------------------------------
    # N-gram modelling
    # ------------------------------------------------------------------

    if ngram == "bigram":
        model = gensim.models.Phrases(all_tokens, min_count=2, threshold=50)
        all_tokens = [model[t] for t in all_tokens]

    elif ngram == "trigram":
        bigram = gensim.models.Phrases(all_tokens, min_count=2, threshold=50)
        trigram = gensim.models.Phrases(
            bigram[all_tokens],
            min_count=2,
            threshold=50
        )
        all_tokens = [trigram[bigram[t]] for t in all_tokens]

    return all_tokens, raw_texts, doc_names
