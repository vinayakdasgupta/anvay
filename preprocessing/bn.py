import os
import logging
import gensim
from nltk.corpus import stopwords

from utils import (
    remove_punctuation,
    custom_bengali_tokenize,
    stem_tokens,
    remove_most_frequent_words
)


def preprocess_bn(
    file_paths,
    remove_stopwords,
    custom_stopwords,
    normaliser,               # Normaliser or None
    normalisation_order,
    percent,
    ngram
):
    """
    Bengali preprocessing.

    Supported normalisers:
    - None
    - Normaliser(key="bn_stem", kind="stem")
    """

    # ------------------------------------------------------------------
    # Validate normaliser
    # ------------------------------------------------------------------

    if normaliser is not None and normaliser.key != "bn_stem":
        raise ValueError(
            f"Unsupported normaliser for Bengali: {normaliser.key}"
        )

    use_stemming = normaliser is not None

    # ------------------------------------------------------------------
    # Build stopword set
    # ------------------------------------------------------------------

    stop_words = set()

    if remove_stopwords:
        stop_words.update(stopwords.words("bengali"))

    if custom_stopwords:
        stop_words.update(custom_stopwords)

    logging.info(
        f"Stopword removal: {'enabled (NLTK Bengali)' if remove_stopwords else 'disabled'}"
    )

    if use_stemming:
        logging.info(
            f"Lexical normalisation: bn_stem "
            f"({normalisation_order.replace('_', ' → ')})"
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
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()

        clean_text = remove_punctuation(raw)
        tokens = custom_bengali_tokenize(clean_text)

        if use_stemming:
            if normalisation_order == "stem_first":
                tokens = stem_tokens([tokens])[0]
                tokens = [t for t in tokens if len(t) > 1 and t not in stop_words]

            elif normalisation_order == "filter_first":
                tokens = [t for t in tokens if len(t) > 1 and t not in stop_words]
                tokens = stem_tokens([tokens])[0]

            else:
                # Defensive fallback: stem → filter
                tokens = stem_tokens([tokens])[0]
                tokens = [t for t in tokens if len(t) > 1 and t not in stop_words]

        else:
            # No normalisation → filter only
            tokens = [t for t in tokens if len(t) > 1 and t not in stop_words]

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
