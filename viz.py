# -*- coding: utf-8 -*-
import os
from bokeh.plotting import figure, output_file, save
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.layouts import column
from bokeh.models.graphs import from_networkx
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import seaborn as sns
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import pyLDAvis
import pyLDAvis.gensim_models
from matplotlib import rcParams
from matplotlib import font_manager
from gensim.models import CoherenceModel


# Get the base directory of the Flask application
base_dir = os.path.abspath(os.path.dirname(__file__))

font_path = os.path.join(base_dir, 'static', 'fonts', 'NotoSansBengali.ttf')

#load fonts into Matplotlib
if os.path.exists(font_path):
    font_manager.fontManager.addfont(font_path)
    bengali_font = font_manager.FontProperties(fname=font_path)
    print("Font added successfully!")
else:
    print("Font file not found at the specified path.")
from matplotlib import rcParams
rcParams['font.family'] = bengali_font.get_name()

def create_bokeh_visualizations(lda_model, corpus, id2word, output_dir="results"):
    """
    Generate custom LDA visualizations using Bokeh.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Generate a scatter plot for topics using PCA
    topic_term_matrix = lda_model.get_topics()  # Shape: (num_topics, num_words)
    pca = PCA(n_components=2)
    topic_coords = pca.fit_transform(topic_term_matrix)  # Reduce to 2D

    scatter_source = ColumnDataSource(data={
        "x": topic_coords[:, 0],
        "y": topic_coords[:, 1],
        "topic": [f"Topic {i}" for i in range(lda_model.num_topics)],
        "keywords": ["\n".join([word for word, _ in lda_model.show_topic(i, topn=10)]) for i in range(lda_model.num_topics)],
    })

    scatter_plot = figure(title="Topic Scatter Plot", tools="pan,box_zoom,reset,hover,save",
                          x_axis_label="PCA Component 1", y_axis_label="PCA Component 2", width=800, height=400)
    scatter_plot.circle("x", "y", size=10, source=scatter_source, color="navy", alpha=0.6)
    hover = scatter_plot.select_one(HoverTool)
    hover.tooltips = [("Topic", "@topic"), ("Top Words", "@keywords")]

    # Save the scatter plot
    scatter_path = os.path.join(output_dir, "topic_scatter.html")
    output_file(scatter_path)
    save(scatter_plot)

    # 2. Generate bar charts for topic-term distributions
    bar_plots = []

    for topic_idx in range(lda_model.num_topics):
        topic_words = lda_model.show_topic(topic_idx, topn=10)
        words = [word for word, _ in topic_words]
        weights = [weight for _, weight in topic_words]

        bar_source = ColumnDataSource(data={"words": words, "weights": weights})

        # Create horizontal bar plot (hbar) instead of vertical bar plot (vbar)
        bar_plot = figure(y_range=words, title=f"Topic {topic_idx} - Top Words", tools="pan,box_zoom,reset,save",
                      width=800, height=400)
        bar_plot.hbar(y="words", height=0.4, right="weights", source=bar_source, color="teal", alpha=0.7)
    
        # Customize axes labels
        bar_plot.xaxis.axis_label = "Weight"
        bar_plot.yaxis.axis_label = "Words"

        bar_plots.append(bar_plot)

    # Save bar charts as a single file
    bar_chart_path = os.path.join(output_dir, "topic_bars.html")
    output_file(bar_chart_path)
    save(column(*bar_plots))

    return {
        "scatter": "topic_scatter.html",
        "bars": "topic_bars.html",
    }

# Generate heatmap visualization
def create_heatmap(lda_model, output_dir="results"):
    
    os.makedirs(output_dir, exist_ok=True)

    # Prepare data for the heatmap
    data = []
    for topic_idx in range(lda_model.num_topics):
        topic_words = lda_model.show_topic(topic_idx, topn=10)
        for word, weight in topic_words:
            data.append({"Topic": f"Topic {topic_idx + 1}", "Word": word, "Weight": weight})

    df = pd.DataFrame(data)

    # Pivot the DataFrame correctly for the heatmap
    heatmap_data = df.pivot(index="Word", columns="Topic", values="Weight")
    
    # Generate the heatmap
    plt.figure(figsize=(8, 5))
    sns.heatmap(heatmap_data, cmap="YlGnBu", annot=True, fmt=".2f", cbar_kws={'label': 'Weight'})
    plt.title("Topic-Word Heatmap")
    plt.ylabel("Words")
    plt.xlabel("Topics")

    # Save the heatmap
    heatmap_path = os.path.join(output_dir, "heatmap.png")
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    return "heatmap.png"

# Generate topic evolution over time
def create_topic_evolution(lda_model, corpus, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)

    # Compute topic distributions over documents
    topic_distributions = [lda_model.get_document_topics(doc) for doc in corpus]
    
    # Aggregate over time (assume one file represents one time period)
    topic_trends = np.zeros((len(topic_distributions), lda_model.num_topics))
    for idx, doc_topics in enumerate(topic_distributions):
        for topic_id, weight in doc_topics:
            topic_trends[idx, topic_id] = weight

    # Plot topic evolution
    plt.figure(figsize=(8, 5))
    for topic_idx in range(lda_model.num_topics):
        plt.plot(topic_trends[:, topic_idx], label=f"Topic {topic_idx}")
    plt.title("Topic Evolution Over Time")
    plt.xlabel("Time")
    plt.ylabel("Topic Proportion")
    plt.legend(loc="upper right")
    evolution_path = os.path.join(output_dir, "topic_evolution.png")
    plt.savefig(evolution_path)
    plt.close()
    return "topic_evolution.png"

# Generate chord diagram (using Bokeh)
def create_chord_diagram(lda_model, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)

    # Extract pairwise similarities between topics
    topic_term_matrix = lda_model.get_topics()
    similarity_matrix = np.dot(topic_term_matrix, topic_term_matrix.T)

    # Create graph for Bokeh visualization
    G = nx.Graph()
    for i in range(len(similarity_matrix)):
        G.add_node(f"Topic {i}")
    for i in range(len(similarity_matrix)):
        for j in range(i + 1, len(similarity_matrix)):
            G.add_edge(f"Topic {i}", f"Topic {j}", weight=similarity_matrix[i, j])

    plot = figure(title="Chord Diagram", width=550, height=500)
    network = from_networkx(G, nx.spring_layout, scale=2, center=(0, 0))
    plot.renderers.append(network)
    chord_path = os.path.join(output_dir, "chord_diagram.html")
    output_file(chord_path)
    save(plot)
    return "chord_diagram.html"

# Generate hierarchical clustering of topics
def create_hierarchical_clustering(lda_model, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)

    # Extract topic-word distributions
    topic_term_matrix = lda_model.get_topics()

    # Perform hierarchical clustering
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0)
    clustering.fit(topic_term_matrix)

    # Plot dendrogram
    from scipy.cluster.hierarchy import dendrogram
    from scipy.cluster.hierarchy import linkage

    linkage_matrix = linkage(topic_term_matrix, method='ward')
    plt.figure(figsize=(8, 5))
    dendrogram(linkage_matrix, labels=[f"Topic {i}" for i in range(lda_model.num_topics)])
    plt.title("Hierarchical Clustering of Topics")
    plt.xlabel("Topic")
    plt.ylabel("Distance")
    clustering_path = os.path.join(output_dir, "hierarchical_clustering.png")
    plt.savefig(clustering_path)
    plt.close()
    return "hierarchical_clustering.png"

# Generate topic distribution per document
def create_topic_distribution_per_document(lda_model, corpus, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)

    # Compute topic distributions
    topic_distributions = [lda_model.get_document_topics(doc) for doc in corpus]
    data = []
    for doc_id, doc_topics in enumerate(topic_distributions):
        for topic_id, weight in doc_topics:
            data.append({"Document": doc_id, "Topic": topic_id, "Weight": weight})

    df = pd.DataFrame(data)
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x="Document", y="Weight", hue="Topic", dodge=True)
    plt.title("Topic Distribution Per Document")
    distribution_path = os.path.join(output_dir, "topic_distribution.png")
    plt.savefig(distribution_path)
    plt.close()
    return "topic_distribution.png"




def create_coherence_bar_chart(lda_model, corpus, id2word, output_dir="results"):
    """
    Generate a bar chart for topic coherence scores.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Compute coherence scores
    coherence_model = CoherenceModel(model=lda_model, texts=corpus, dictionary=id2word, coherence='c_v')
    coherence_scores = coherence_model.get_coherence_per_topic()

    if not coherence_scores:
        raise ValueError("Coherence scores are empty. Ensure valid topics are generated.")

    # Plot the coherence scores
    plt.figure(figsize=(8, 5))
    plt.bar(range(len(coherence_scores)), coherence_scores, color="skyblue", alpha=0.7)
    plt.xlabel("Topics")
    plt.ylabel("Coherence Score")
    plt.title("Topic Coherence Scores")
    coherence_path = os.path.join(output_dir, "coherence_bar_chart.png")
    plt.savefig(coherence_path)
    plt.close()
    return "coherence_bar_chart.png"

