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


def preprocess_documents(
    file_paths,
    remove_stopwords,
    custom_stopwords,
    use_stemming,
    normalisation_order,
    percent,
    ngram,
    language="bn"
):
    # Build stopwords set
    stop_words = set(stopwords.words('bengali')) if remove_stopwords else set()
    if custom_stopwords:
        stop_words.update(custom_stopwords)

    logging.info(f"Normalisation order: {normalisation_order}")

    all_tokens, raw_texts, doc_names = [], [], []

    for path in file_paths:
        with open(path, 'r', encoding='utf-8') as f:
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
                # safe fallback: stem_first behaviour
                tokens = stem_tokens([tokens])[0]
                tokens = [t for t in tokens if len(t) > 1 and t not in stop_words]

        else:
            # stemming disabled → always filter only
            tokens = [t for t in tokens if len(t) > 1 and t not in stop_words]

        all_tokens.append(tokens)
        raw_texts.append(clean_text)
        doc_names.append(os.path.basename(path))

    # Optionally remove most frequent words
    if percent > 0:
        all_tokens = remove_most_frequent_words(all_tokens, percent)

    # Build bigrams/trigrams if requested
    if ngram == 'bigram':
        model = gensim.models.Phrases(all_tokens, min_count=2, threshold=50)
        all_tokens = [model[t] for t in all_tokens]

    elif ngram == 'trigram':
        bigram = gensim.models.Phrases(all_tokens, min_count=2, threshold=50)
        trigram = gensim.models.Phrases(bigram[all_tokens], min_count=2, threshold=50)
        all_tokens = [trigram[bigram[t]] for t in all_tokens]

    return all_tokens, raw_texts, doc_names
