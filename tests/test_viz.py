# tests/test_viz.py

import pytest
import numpy as np
from gensim.corpora import Dictionary
from gensim.models import LdaModel

from viz import (
    create_interactive_scatter,
    create_interactive_bar_charts,
    create_interactive_heatmap,
    create_interactive_topic_evolution,
    create_interactive_topic_distribution,
    create_interactive_clustering,
    create_topic_prevalence_pie,
    create_topic_word_network,
    prepare_topic_doc_drilldown,
    get_representative_sentences_custom,
    create_corpus_top_tokens_bar
)

# Build a tiny dummy LDA model and corpus
@pytest.fixture(scope="module")
def dummy_lda_model():
    texts = [["আমি", "তুমি", "সে"], ["বই", "পড়া", "পছন্দ"], ["নদী", "বয়ে", "চলে"]]
    dictionary = Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=2, passes=5)
    return lda_model, corpus, dictionary, texts

# ---------- Plotly-based HTML outputs ----------
def test_scatter_returns_html(dummy_lda_model):
    html = create_interactive_scatter(dummy_lda_model[0])
    assert "<div" in html

def test_bar_charts_html(dummy_lda_model):
    html = create_interactive_bar_charts(dummy_lda_model[0])
    assert "<div" in html

def test_heatmap_html(dummy_lda_model):
    html = create_interactive_heatmap(dummy_lda_model[0])
    assert "<div" in html

def test_evolution_plot_html(dummy_lda_model):
    html = create_interactive_topic_evolution(dummy_lda_model[0], dummy_lda_model[1])
    assert "<div" in html

def test_distribution_plot_html(dummy_lda_model):
    html = create_interactive_topic_distribution(dummy_lda_model[0], dummy_lda_model[1])
    assert "<div" in html

def test_clustering_plot_html(dummy_lda_model):
    html = create_interactive_clustering(dummy_lda_model[0])
    assert "<div" in html

def test_prevalence_pie_html(dummy_lda_model):
    html = create_topic_prevalence_pie(dummy_lda_model[0], dummy_lda_model[1])
    assert "<div" in html

def test_topic_word_network_html(dummy_lda_model):
    html = create_topic_word_network(dummy_lda_model[0])
    assert "<div" in html

def test_top_tokens_bar_html():
    tokens = [("আমি", 10), ("তুমি", 8), ("সে", 5)]
    html = create_corpus_top_tokens_bar(tokens)
    assert "<div" in html

# ---------- Sentence/paragraph-based methods ----------
def test_prepare_topic_doc_drilldown(dummy_lda_model):
    lda, corpus, _, texts = dummy_lda_model
    raw_texts = [" ".join(t) for t in texts]
    doc_names = [f"doc{i}" for i in range(len(texts))]

    data = prepare_topic_doc_drilldown(
        lda_model=lda,
        corpus=corpus,
        doc_names=doc_names,
        raw_texts=raw_texts,
        min_weight=0.0
    )
    assert isinstance(data, dict)
    assert all(isinstance(v, list) for v in data.values())

def test_get_representative_sentences_custom(dummy_lda_model):
    lda, corpus, dictionary, texts = dummy_lda_model
    raw_texts = [" ".join(t) for t in texts]
    doc_names = [f"doc{i}" for i in range(len(texts))]

    result = get_representative_sentences_custom(
        lda_model=lda,
        corpus=corpus,
        raw_texts=raw_texts,
        dictionary=dictionary,
        doc_names=doc_names,
        num_topics=2
    )
    assert isinstance(result, dict)
    assert len(result) == 2
    for topic_id, entry in result.items():
        assert "doc" in entry
        assert "text" in entry
        assert "weight" in entry
