import os
import gensim
from contextlib import redirect_stdout


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
    """

    with redirect_stdout(log_stream):
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
