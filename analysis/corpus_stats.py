# analysis/corpus_stats.py

from collections import Counter
import numpy as np


def compute_corpus_stats(all_tokens, normalisation_order):
    """
    Compute corpus-level statistics from tokenized documents.

    Parameters
    ----------
    all_tokens : list[list[str]]
        Tokenized documents after preprocessing.
    normalisation_order : str
        Normalisation order used ("stem_first", "filter_first", etc.).

    Returns
    -------
    overview_stats : dict
        High-level corpus statistics for display.
    top_tokens : list[tuple]
        Top tokens with raw counts.
    top_token_text : list[dict]
        Top tokens with percentage values for visualization.
    """

    total_docs = len(all_tokens)

    flat_tokens = [t for doc in all_tokens for t in doc]
    total_tokens = len(flat_tokens)
    vocab_size = len(set(flat_tokens))

    avg_len = round(np.mean([len(doc) for doc in all_tokens]), 2)

    top_tokens = Counter(flat_tokens).most_common(10)

    top_token_text = [
        {
            "token": token,
            "percent": round(count / total_tokens * 100, 2)
        }
        for token, count in top_tokens
    ]

    overview_stats = {
        "Total Documents": total_docs,
        "Total Tokens": total_tokens,
        "Vocabulary Size": vocab_size,
        "Average Document Length": avg_len,
        "Shortest Document Length": min(len(doc) for doc in all_tokens),
        "Longest Document Length": max(len(doc) for doc in all_tokens),
        "Normalisation Order": normalisation_order
    }

    return overview_stats, top_tokens, top_token_text
