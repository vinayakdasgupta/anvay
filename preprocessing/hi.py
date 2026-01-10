# -*- coding: utf-8 -*-

import os
import logging
import re
import string
import gensim

import simplemma

from utils import remove_most_frequent_words


# ------------------------------------------------------------------
# Hindi Unicode + punctuation setup (mirrors Bengali logic)
# ------------------------------------------------------------------

DEVANAGARI_UNICODE_RANGE = r'\u0900-\u097F'
ZERO_WIDTH = r'\u200B\u200C\u200D\u200E\u200F\u202A-\u202E\u2060\uFEFF'
CUSTOM_PUNCTUATION = string.punctuation + '।॥“”‘’—–…•·―'


def custom_hindi_tokenize(text):
    """
    Tokeniser modelled exactly on custom_bengali_tokenize,
    adapted for Devanagari.
    """

    # Remove punctuation and zero-width chars
    text = re.sub(
        f"[{re.escape(CUSTOM_PUNCTUATION)}{ZERO_WIDTH}]",
        " ",
        text
    )

    # Retain only Devanagari characters and spaces
    cleaned = "".join(
        ch for ch in text
        if ch == " " or "\u0900" <= ch <= "\u097F"
    )

    tokens = cleaned.split()

    # Final sanity filter (paranoia, same as Bengali)
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

def _get_hindi_stopwords():
    """
    Hindi stopwords are handled via custom lists only.
    NLTK does not provide Hindi stopwords.
    """
    return set()


# ------------------------------------------------------------------
# Main preprocessing function
# ------------------------------------------------------------------

def preprocess_hi(
    file_paths,
    remove_stopwords,
    custom_stopwords,
    normaliser,          # Normaliser or None
    percent,
    ngram
):
    """
    Hindi preprocessing.

    Tokenisation:
    - Custom Devanagari tokenizer (Bengali-style logic)

    Normalisation:
    - hi_lemma via simplemma (dictionary-based, experimental)

    Order:
    - filter → lemmatize (fixed)
    """

    # ------------------------------------------------------------------
    # Validate normaliser
    # ------------------------------------------------------------------

    if normaliser is not None and normaliser.key != "hi_lemma":
        raise ValueError(
            f"Unsupported normaliser for Hindi: {normaliser.key}"
        )

    use_lemmatization = normaliser is not None

    # ------------------------------------------------------------------
    # Build stopword set
    # ------------------------------------------------------------------

    stop_words = set()

    if remove_stopwords:
        stop_words.update(_get_hindi_stopwords())

    if custom_stopwords:
        stop_words.update(custom_stopwords)

    logging.info(
        "Hindi tokenisation: custom Devanagari tokenizer"
    )

    if use_lemmatization:
        logging.info(
            "Lexical normalisation: hi_lemma (simplemma, fixed order)"
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

        tokens = custom_hindi_tokenize(raw)

        # Filter first
        filtered = [
            t for t in tokens
            if len(t) > 1 and t not in stop_words
        ]

        # Optional lemmatisation
        if use_lemmatization:
            tokens = [
                simplemma.lemmatize(t, lang="hi")
                for t in filtered
            ]
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
