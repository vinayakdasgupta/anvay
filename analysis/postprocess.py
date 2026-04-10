# analysis/postprocess.py
from utils import get_relevance_weighted_words, generate_topic_labels
from viz import get_representative_sentences_custom


def compute_topic_semantics(
    lda_model,
    corpus,
    id2word,
    raw_texts,
    doc_names,
    language,
    lambda_val=0.6
):
    """
    Compute semantic enrichments for topics:
    - relevance-weighted words
    - topic labels
    - representative sentences
    """

    relevance_topics = get_relevance_weighted_words(
        lda_model, corpus, id2word, lambda_val=lambda_val
    )

    topic_labels = generate_topic_labels(relevance_topics)

    representative_sents = get_representative_sentences_custom(
        lda_model,
        corpus,
        raw_texts,
        id2word,
        doc_names=doc_names,
        language=language
    )

    return relevance_topics, topic_labels, representative_sents
