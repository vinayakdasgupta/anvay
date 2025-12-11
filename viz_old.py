# viz.py (updated with Plotly and HTML embedding for Flask integration)

import os
import numpy as np
import pandas as pd
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
from collections import Counter
import re
from utils import (
    custom_bengali_tokenize,
    split_sentences_bengali
)

# Utility to export HTML divs (no full HTML)
def plot_to_div(fig):
    return pio.to_html(fig, full_html=False, include_plotlyjs='cdn')

def create_interactive_scatter(lda_model):
    topic_term_matrix = lda_model.get_topics()
    pca = PCA(n_components=2)
    coords = pca.fit_transform(topic_term_matrix)

    fig = px.scatter(
        x=coords[:, 0],
        y=coords[:, 1],
        text=[f"Topic {i}" for i in range(len(coords))],
        hover_name=[f"Topic {i}" for i in range(len(coords))],
        labels={'x': 'PCA 1', 'y': 'PCA 2'}
    )
    fig.update_traces(marker=dict(size=12, color='indianred'), textposition='top center')
    fig.update_layout(margin=dict(l=20, r=20, t=60, b=20))
    return plot_to_div(fig)

def create_interactive_bar_charts(lda_model):
    data = []
    for topic_id in range(lda_model.num_topics):
        topic_words = lda_model.show_topic(topic_id, topn=10)
        for word, weight in topic_words:
            data.append({"Topic": f"Topic {topic_id}", "Word": word, "Weight": weight})

    df = pd.DataFrame(data)
    fig = px.bar(df, x='Weight', y='Word', color='Topic', orientation='h', height=600)
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    return plot_to_div(fig)

def create_interactive_heatmap(lda_model):
    data = []
    for topic_id in range(lda_model.num_topics):
        for word, weight in lda_model.show_topic(topic_id, topn=10):
            data.append({"Topic": f"Topic {topic_id}", "Word": word, "Weight": weight})

    df = pd.DataFrame(data)
    pivot_df = df.pivot(index="Word", columns="Topic", values="Weight").fillna(0)
    fig = px.imshow(pivot_df, labels=dict(x="Topic", y="Word", color="Weight"))
    return plot_to_div(fig)

def create_interactive_topic_evolution(lda_model, corpus):
    topic_matrix = np.zeros((len(corpus), lda_model.num_topics))
    for i, doc in enumerate(corpus):
        for topic_id, weight in lda_model.get_document_topics(doc):
            topic_matrix[i, topic_id] = weight

    df = pd.DataFrame(topic_matrix, columns=[f"Topic {i}" for i in range(lda_model.num_topics)])
    df["Time"] = range(len(df))
    fig = px.line(df, x="Time", y=df.columns[:-1])
    return plot_to_div(fig)

def create_interactive_topic_distribution(lda_model, corpus):
    data = []
    for doc_id, doc in enumerate(corpus):
        for topic_id, weight in lda_model.get_document_topics(doc):
            data.append({"Document": f"Doc {doc_id}", "Topic": f"Topic {topic_id}", "Weight": weight})

    df = pd.DataFrame(data)
    fig = px.bar(df, x="Document", y="Weight", color="Topic", height=600)
    return plot_to_div(fig)

def create_interactive_clustering(lda_model):
    topic_term_matrix = lda_model.get_topics()
    Z = linkage(topic_term_matrix, method='ward')
    dendro = dendrogram(Z, no_plot=True)

    labels = [f"Topic {i}" for i in dendro['leaves']]
    icoord = np.array(dendro['icoord'])
    dcoord = np.array(dendro['dcoord'])

    fig = go.Figure()
    for i in range(len(icoord)):
        x = icoord[i]
        y = dcoord[i]
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line=dict(color='black')))

    fig.update_layout(
        xaxis=dict(tickvals=[5 + 10 * i for i in range(len(labels))], ticktext=labels),
        yaxis_title='Distance',
        plot_bgcolor='white', paper_bgcolor='white', font=dict(color='black'),
        margin=dict(l=20, r=20, t=60, b=20)
    )
    return plot_to_div(fig)

def create_topic_prevalence_pie(lda_model, corpus):
    topic_totals = [0.0] * lda_model.num_topics
    for doc in corpus:
        for topic_id, weight in lda_model.get_document_topics(doc):
            topic_totals[topic_id] += weight

    df = pd.DataFrame({
        "Topic": [f"Topic {i}" for i in range(lda_model.num_topics)],
        "Weight": topic_totals
    })

    fig = px.pie(df, names="Topic", values="Weight")
    return plot_to_div(fig)

def create_topic_word_network(lda_model, topn=10):
    G = nx.Graph()
    for topic_id in range(lda_model.num_topics):
        topic_label = f"Topic {topic_id}"
        G.add_node(topic_label, type='topic')
        for word, weight in lda_model.show_topic(topic_id, topn=topn):
            G.add_node(word, type='word')
            G.add_edge(topic_label, word, weight=weight)

    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')

    node_x, node_y, node_text, node_color = [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        node_color.append('#0D85D8' if G.nodes[node]['type'] == 'topic' else '#888888')

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers+text', hoverinfo='text',
        marker=dict(size=12, color=node_color), text=node_text, textposition="bottom center")

    fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(
        showlegend=False, hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40), paper_bgcolor='white', plot_bgcolor='white'))

    return plot_to_div(fig)

def prepare_topic_doc_drilldown(lda_model, corpus, doc_names=None, raw_texts=None, min_weight=0.2):
    """
    Returns up to 5 unique paragraphs per topic.
    Each entry includes real doc label, paragraph text, and topic weight.
    """
    from collections import defaultdict

    topic_to_paragraphs = defaultdict(list)
    seen = defaultdict(set)

    for doc_id, bow in enumerate(corpus):
        topic_dist = lda_model.get_document_topics(bow)
        for topic_id, weight in topic_dist:
            if weight < min_weight:
                continue

            paragraph_text = raw_texts[doc_id].strip()
            paragraph_id = f"{doc_names[doc_id]}" if doc_names else f"Doc {doc_id}"

            # Avoid repeating same paragraph under same topic
            key = (paragraph_id, paragraph_text)
            if key in seen[topic_id]:
                continue
            seen[topic_id].add(key)

            topic_to_paragraphs[f"Topic {topic_id}"].append({
                "doc": paragraph_id,
                "weight": round(weight, 4),
                "text": paragraph_text
            })

    # Sort each topic group and keep top 5 unique paragraphs
    for topic_id in topic_to_paragraphs:
        topic_to_paragraphs[topic_id] = sorted(
            topic_to_paragraphs[topic_id],
            key=lambda x: -x["weight"]
        )[:5]

    return dict(topic_to_paragraphs)









def create_corpus_top_tokens_bar(top_tokens):
    df = pd.DataFrame(top_tokens, columns=["Token", "Frequency"])
    fig = px.bar(df, x="Token", y="Frequency", title="Top 10 Frequent Tokens in Corpus")
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    return pio.to_html(fig, full_html=False, include_plotlyjs='cdn')







def get_representative_sentences_custom(lda_model, corpus, raw_texts, dictionary, doc_names=None, num_topics=10, topn=10):
    topic_sentences = {}
    seen_docs = set()

    for topic_id in range(num_topics):
        max_score = 0
        best_doc_index = -1

        # Find highest-weight doc for topic not already used
        for i, bow in enumerate(corpus):
            if i in seen_docs:
                continue
            topic_dist = dict(lda_model.get_document_topics(bow))
            score = topic_dist.get(topic_id, 0)
            if score > max_score:
                max_score = score
                best_doc_index = i

        if best_doc_index != -1:
            seen_docs.add(best_doc_index)
            raw_text = raw_texts[best_doc_index]
            sentences = split_sentences_bengali(raw_text)
            top_words = [dictionary[w_id] for w_id, _ in lda_model.get_topic_terms(topic_id, topn=topn)]


            best_sent = sentences[0] if sentences else ""
            best_overlap = -1
            for sent in sentences:
                tokens = custom_bengali_tokenize(sent)
                overlap = len(set(tokens) & set(top_words))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_sent = sent
                    

            topic_sentences[topic_id] = {
                "doc": doc_names[best_doc_index] if doc_names else f"Doc {best_doc_index}",
                "weight": round(max_score, 4),
                "text": best_sent,
                "low_confidence": best_overlap == 0  

            }

    return topic_sentences
