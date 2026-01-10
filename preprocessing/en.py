import os
import logging
import gensim
import re

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from utils import remove_most_frequent_words


_lemmatizer = WordNetLemmatizer()
_stopwords_en = set(stopwords.words("english"))


def preprocess_en(
    file_paths,
    remove_stopwords,
    custom_stopwords,
    normaliser,              # Normaliser or None
    percent,
    ngram
):
    """
    English preprocessing.

    Supported normalisers:
    - None
    - Normaliser(key="en_lemma", kind="lemma")

    Notes:
    - Normalisation order is fixed: filter → lemmatize
    - normalisation_order is intentionally not supported
    """

    # ------------------------------------------------------------------
    # Validate normaliser
    # ------------------------------------------------------------------

    if normaliser is not None and normaliser.key != "en_lemma":
        raise ValueError(
            f"Unsupported normaliser for English: {normaliser.key}"
        )

    use_lemmatization = normaliser is not None

    # ------------------------------------------------------------------
    # Build stopword set
    # ------------------------------------------------------------------

    stop_words = set()

    if remove_stopwords:
        stop_words.update(_stopwords_en)

    if custom_stopwords:
        stop_words.update(w.lower() for w in custom_stopwords)

    logging.info(
        f"Stopword removal: {'enabled (NLTK English)' if remove_stopwords else 'disabled'}"
    )

    if use_lemmatization:
        logging.info(
            "Lexical normalisation: en_lemma (filter → lemmatize, fixed)"
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

        # Simple punctuation cleanup
        clean_text = re.sub(r"[^\w\s]", " ", raw)

        tokens = word_tokenize(clean_text)

        # Filter first (always)
        filtered = [
            t for t in tokens
            if len(t) > 1 and t.isalpha() and t not in stop_words
        ]

        # Optional lemmatization
        if use_lemmatization:
            tokens = [_lemmatizer.lemmatize(t) for t in filtered]
        else:
            tokens = filtered

        all_tokens.append(tokens)
        raw_texts.append(clean_text)
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
        trigram = gensim.models.Phrases(bigram[all_tokens], min_count=2, threshold=50)
        all_tokens = [trigram[bigram[t]] for t in all_tokens]

    return all_tokens, raw_texts, doc_names
