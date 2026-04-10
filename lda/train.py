import os
import gensim


def train_lda_model(
    corpus,
    id2word,
    num_topics,
    iterations,
    passes,
    chunk_size,
    alpha,
    eta,
    per_word_topics,
    minimum_probability,
    use_multicore,
    log_stream
):
    """
    Train and return a Gensim LDA model.

    Note: log_stream is kept in the signature for API compatibility.
    Actual log capture is handled in the caller via a logging.StreamHandler
    attached to the 'gensim' logger — Gensim emits via logging, not stdout.
    """
    lda_model = gensim.models.LdaMulticore(
        corpus=corpus,
        id2word=id2word,
        num_topics=num_topics,
        iterations=iterations,
        passes=passes,
        chunksize=chunk_size,
        alpha=alpha,
        eta=eta,
        per_word_topics=per_word_topics,
        minimum_probability=minimum_probability,
        workers=os.cpu_count() - 1 if use_multicore else 1
    )

    return lda_model
