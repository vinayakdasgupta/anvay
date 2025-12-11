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
import json
import uuid
import html
from utils import (
    custom_bengali_tokenize,
    split_sentences_bengali
)

# ------------------ Style / Theme constants (central) ------------------
# Consistent font family (site font + Bengali shaping fallback)
FONT_FAMILY = "Roboto, 'Noto Sans Bengali', 'Noto Sans', sans-serif"
FONT_SIZE = 11

# Brand/topic color palette:
# Start with brand colour then fall back to Plotly qualitative palette.
BRAND_COLOR = "#0D85D8"
DEFAULT_PALETTE = px.colors.qualitative.Plotly  # reliable qualitative palette

def build_topic_color_map(num_topics):
    """
    Return a dict mapping 'Topic 0'..'Topic N' -> hex colour.
    Ensures Topic 0 uses BRAND_COLOR and cycles the default palette for others.
    """
    # create working palette copy
    palette = DEFAULT_PALETTE.copy()
    # ensure first colour is brand colour
    if len(palette) > 0:
        palette[0] = BRAND_COLOR
    # extend palette if num_topics exceeds length by repeating
    color_list = [palette[i % len(palette)] for i in range(num_topics)]
    return {f"Topic {i}": color_list[i] for i in range(num_topics)}
# -----------------------------------------------------------------------


def get_topic_tooltips(lda_model, topn=5):
    """
    Return a dict mapping topic_id -> short string of topn words (joined by ', ').
    """
    tooltips = {}
    # robustly obtain number of topics
    num_topics = getattr(lda_model, "num_topics", None)
    if num_topics is None:
        # fallback for older gensim: use shape of topic matrix
        try:
            num_topics = lda_model.get_topics().shape[0]
        except Exception:
            num_topics = 0
    for tid in range(num_topics):
        try:
            words = [w for w, _ in lda_model.show_topic(tid, topn=topn)]
        except Exception:
            # fallback to show_topic default behaviour
            words = [w for w, _ in lda_model.show_topic(tid, topn=topn)]
        tooltips[tid] = ", ".join(words)
    return tooltips

# Utility to export HTML divs (no full HTML)
def plot_to_div(fig):
    return pio.to_html(fig, full_html=False, include_plotlyjs='cdn')

def create_interactive_scatter(lda_model):
    topic_term_matrix = lda_model.get_topics()
    pca = PCA(n_components=2)
    coords = pca.fit_transform(topic_term_matrix)

    tooltips = get_topic_tooltips(lda_model, topn=5)
    labels = [f"Topic {i}" for i in range(len(coords))]

    df = pd.DataFrame({
        "x": coords[:, 0],
        "y": coords[:, 1],
        "label": labels,
        "top_words": [tooltips[i] for i in range(len(coords))]
    })

    color_map = build_topic_color_map(len(coords))

    # Tell px to colour by the label column and use our map
    fig = px.scatter(
        df,
        x="x",
        y="y",
        text="label",
        color="label",                      # <- important
        color_discrete_map=color_map,
        height=480
    )
    fig.update_layout(font=dict(family=FONT_FAMILY, size=FONT_SIZE))

    # only set marker size here — do NOT set marker.color
    fig.update_traces(
        marker=dict(size=14),
        textposition='top center',
        hovertemplate="<b>%{text}</b><br>Top words: %{customdata[0]}<extra></extra>",
        customdata=df[["top_words"]].values
    )

    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    return plot_to_div(fig)


def create_interactive_bar_charts(lda_model):
    tooltips = get_topic_tooltips(lda_model, topn=5)

    data = []
    for topic_id in range(lda_model.num_topics):
        topic_words = lda_model.show_topic(topic_id, topn=10)
        for word, weight in topic_words:
            data.append({
                "Topic": f"Topic {topic_id}",
                "Word": word,
                "Weight": weight,
                "top_words": tooltips[topic_id]
            })

    # --- start replacement block ---
    df = pd.DataFrame(data)

# set how many global words to show
    GLOBAL_TOP_N = 25  # change to 10-15 as you prefer

# 1) choose the top GLOBAL_TOP_N words by total weight across topics
    global_top_words = (
        df.groupby("Word")["Weight"]
        .sum()
        .nlargest(GLOBAL_TOP_N)
        .index
        .tolist()
    )

# 2) build a complete topic x word matrix for the selected words.
#    This ensures each (Topic, Word) pair exists (weight 0 if absent), so every
#    topic is represented for every selected global word.
    pivot = (
        df[df["Word"].isin(global_top_words)]
        .pivot_table(index="Word", columns="Topic", values="Weight", aggfunc="sum")
        .reindex(index=global_top_words)   # preserve the chosen ordering
    )

# ensure all topics are present as columns (even if some have no weight for these words)
    all_topics = [f"Topic {i}" for i in range(lda_model.num_topics)]
    for t in all_topics:
        if t not in pivot.columns:
            pivot[t] = 0.0

# Reorder columns to Topic 0..Topic N
    pivot = pivot[all_topics]

# 3) melt back to long form for Plotly, filling missing with 0
    df_long = pivot.reset_index().melt(id_vars="Word", var_name="Topic", value_name="Weight")
    df_long["Weight"] = df_long["Weight"].fillna(0.0)

# 4) preserve ordering for the y-axis (largest on top visually)
    df_long["Word"] = pd.Categorical(df_long["Word"],
                                 categories=global_top_words,
                                 ordered=True)

# 5) use the original 'top_words' tooltip strings for topics (unchanged)
    tooltips = get_topic_tooltips(lda_model, topn=5)
# attach tooltip text per Topic so hover remains identical to previous behaviour
    df_long["top_words"] = df_long["Topic"].apply(lambda t: tooltips.get(int(t.replace("Topic ", "")), ""))

# 6) final figure (no theme or hover template changes from your code)
    color_map = build_topic_color_map(lda_model.num_topics)
    fig = px.bar(df_long, x='Weight', y='Word', color='Topic', orientation='h',
             height=600, color_discrete_map=color_map, category_orders={'Word': global_top_words})
    fig.update_yaxes(autorange="reversed")

# reattach your hover template exactly as before (keeping your tooltips)
    fig.update_traces(
        customdata=df_long[["top_words", "Topic"]].values,
        hovertemplate="<b>%{y}</b><br>Topic: %{customdata[1]}<br>Weight: %{x}<br>Top words: %{customdata[0]}<extra></extra>"
    )
    fig.update_layout(yaxis={'categoryorder': 'array',
                         'categoryarray': global_top_words,})
    fig.update_layout(font=dict(family=FONT_FAMILY, size=FONT_SIZE))

# --- end replacement block ---

   

    return plot_to_div(fig)



def create_interactive_heatmap(lda_model):
    topn = 10

    # --- STEP 1: collect per-topic top words (unchanged) ---
    data = []
    for topic_id in range(lda_model.num_topics):
        for word, weight in lda_model.show_topic(topic_id, topn=topn):
            data.append({
                "Topic": f"Topic {topic_id}",
                "Word": word,
                "Weight": weight,
                "topic_id": topic_id
            })

    df = pd.DataFrame(data)

    # --- STEP 2: GLOBAL TOP-N selection (same logic as bar chart) ---
    GLOBAL_TOP_N = 20   # set 10–15–25 as you prefer

    global_top_words = (
        df.groupby("Word")["Weight"]
          .sum()
          .nlargest(GLOBAL_TOP_N)
          .index
          .tolist()
    )

    # --- STEP 3: build topic × word matrix ONLY for selected global words ---
    pivot_df = (
        df[df["Word"].isin(global_top_words)]
          .pivot_table(index="Word", columns="Topic", values="Weight", aggfunc="sum")
          .reindex(index=global_top_words)        # preserve ordering: highest → lowest
          .fillna(0)
    )

    # ensure all topics appear as columns (even if empty)
    all_topics = [f"Topic {i}" for i in range(lda_model.num_topics)]
    for t in all_topics:
        if t not in pivot_df.columns:
            pivot_df[t] = 0.0
    pivot_df = pivot_df[all_topics]

    # --- STEP 4: prepare arrays for heatmap ---
    z = pivot_df.values
    words = pivot_df.index.tolist()      # already ordered from highest → lowest importance
    topics = pivot_df.columns.tolist()

    # --- STEP 5: rebuild tooltip matrix (your original logic, unchanged) ---
    tooltips = get_topic_tooltips(lda_model, topn=5)
    customdata = []
    for r in range(len(words)):
        row_cd = []
        for col in topics:
            tid = int(col.replace("Topic ", ""))
            row_cd.append(tooltips.get(tid, ""))
        customdata.append(row_cd)

    # --- STEP 6: heatmap (your original config) ---
    heat = go.Heatmap(
        z=z,
        x=topics,
        y=words,
        customdata=customdata,
        hovertemplate=(
            "Word: %{y}<br>"
            "Topic: %{x}<br>"
            "Weight: %{z}<br>"
            "Top words: %{customdata}<extra></extra>"
        ),
        colorbar=dict(title="Weight")
    )

    fig = go.Figure(data=[heat])

    # ORDERING FIX → put highest-weight words at top
    fig.update_yaxes(autorange="reversed")

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis_title="Topic",
        yaxis_title="Word"
    )
    fig.update_layout(font=dict(family=FONT_FAMILY, size=FONT_SIZE))

    return plot_to_div(fig)



def create_interactive_topic_evolution(lda_model, corpus, doc_names=None, debug=False):
    import pandas as pd
    import plotly.express as px
    import numpy as np

    # build topic matrix by document index
    topic_matrix = np.zeros((len(corpus), lda_model.num_topics))
    for i, doc in enumerate(corpus):
        for topic_id, weight in lda_model.get_document_topics(doc):
            topic_matrix[i, topic_id] = weight

    df = pd.DataFrame(topic_matrix, columns=[f"Topic {i}" for i in range(lda_model.num_topics)])
    df["DocumentIndex"] = range(len(df))

    # build document name mapping (safe)
    if doc_names is not None and len(doc_names) == len(df):
        df["Document"] = doc_names
    else:
        # fallback labels
        df["Document"] = df["DocumentIndex"].apply(lambda i: f"Doc {i}")

    # mapping from DocumentIndex -> Document (strings)
    index_to_doc = dict(zip(df["DocumentIndex"].astype(int).tolist(), df["Document"].astype(str).tolist()))

    long = df.melt(id_vars=["DocumentIndex", "Document"], var_name="Topic", value_name="Weight")
    tooltips = get_topic_tooltips(lda_model, topn=5)
    long["top_words"] = long["Topic"].apply(lambda t: tooltips[int(t.replace("Topic ", ""))])

    # create the line plot (one trace per topic)
    color_map = build_topic_color_map(lda_model.num_topics)
    fig = px.line(long, x="DocumentIndex", y="Weight", color="Topic", height=480, color_discrete_map=color_map)
    fig.update_layout(font=dict(family=FONT_FAMILY, size=FONT_SIZE))

    # attach per-trace customdata aligned with trace.x values robustly:
    for trace in fig.data:
        try:
            tid = int(trace.name.replace("Topic ", ""))
            # trace.x may be numpy array of floats; convert to ints safely
            x_vals = np.array(trace.x)
            # attempt to coerce to integer indices
            try:
                doc_indices = [int(round(float(x))) for x in x_vals]
            except Exception:
                # fallback: try direct int conversion
                doc_indices = []
                for x in x_vals:
                    try:
                        doc_indices.append(int(x))
                    except Exception:
                        doc_indices.append(None)

            # map indices -> document names using our mapping; fallback to "Doc {i}"
            doc_names_for_trace = []
            for idx in doc_indices:
                if idx is None:
                    doc_names_for_trace.append("")   # missing value
                else:
                    doc_names_for_trace.append(index_to_doc.get(int(idx), f"Doc {idx}"))

            # build topwords repeated array (one per point)
            topwords_for_trace = [tooltips.get(tid, "")] * len(doc_names_for_trace)

            # stack to customdata shape (N,2)
            custom = np.column_stack([np.array(doc_names_for_trace, dtype=object), np.array(topwords_for_trace, dtype=object)])
            trace.customdata = custom
            # hover includes filename as requested
            trace.hovertemplate = (
                "<b>%{fullData.name}</b><br>"
                "Document: %{customdata[0]}<br>"
                "Document index: %{x}<br>"
                "Weight: %{y}<br>"
                "Top words: %{customdata[1]}<extra></extra>"
            )

            if debug:
                print(f"[topic_evolution] trace.name={trace.name}, len(trace.x)={len(trace.x)}, sample x_vals={trace.x[:5]}")
                print(f"[topic_evolution] doc_names_for_trace sample={doc_names_for_trace[:5]}")

        except Exception as e:
            if debug:
                print("Error attaching customdata for trace:", getattr(trace, "name", None), str(e))
            # leave default if parsing fails
            pass

    fig.update_layout(xaxis_title="Document index (upload order)", margin=dict(l=0, r=0, t=0, b=0))
    return plot_to_div(fig)



def create_interactive_topic_distribution(lda_model, corpus, doc_names=None):
    import pandas as pd
    import plotly.express as px
    import numpy as np
    import os

    data = []

    # Build safe doc name mapping
    N = len(corpus)
    if doc_names is not None and len(doc_names) == N:
        # Use basenames to avoid long paths
        safe_doc_names = [os.path.basename(str(x)) for x in doc_names]
    else:
        safe_doc_names = [f"Doc {i}" for i in range(N)]

    # Populate rows with filenames instead of Doc i
    for doc_id, doc in enumerate(corpus):
        doc_label = safe_doc_names[doc_id]
        for topic_id, weight in lda_model.get_document_topics(doc):
            data.append({
                "Document": doc_label,
                "Topic": f"Topic {topic_id}",
                "Weight": weight
            })

    df = pd.DataFrame(data)
    tooltips = get_topic_tooltips(lda_model, topn=5)
    color_map = build_topic_color_map(lda_model.num_topics)
    fig = px.bar(df, x='Document', y='Weight', color='Topic', height=600, color_discrete_map=color_map)
    fig.update_layout(font=dict(family=FONT_FAMILY, size=FONT_SIZE))

    # Attach customdata (top words) and hovertemplate
    for trace in fig.data:
        try:
            tid = int(trace.name.replace("Topic ", ""))
            trace.customdata = np.repeat(tooltips.get(tid, ""), len(trace.x)).reshape(-1, 1)
            trace.hovertemplate = (
                "<b>%{x}</b><br>"
                "Topic: " + trace.name + "<br>"
                "Weight: %{y}<br>"
                "Top words: %{customdata[0]}<extra></extra>"
            )
        except Exception:
            continue

    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    return plot_to_div(fig)



def create_interactive_clustering(lda_model):
    """
    Improved hierarchical clustering with:
      - cluster-level keyword summaries (BERTopic-style)
      - hover tooltips for all merges
      - consistent colour scheme applied to branch lines, merge nodes, and topic labels
    Behaviour:
      - Each merge (internal node) is coloured according to a representative topic
        computed from the merged member topics (majority / highest-sum).
    """
    import numpy as np
    import plotly.graph_objects as go
    from scipy.cluster.hierarchy import linkage, dendrogram
    from collections import Counter

    topic_term_matrix = lda_model.get_topics()
    # lda_model.id2word may be a gensim Dictionary-like; fallback robustly
    try:
        vocab = lda_model.id2word
    except Exception:
        vocab = {i: str(i) for i in range(topic_term_matrix.shape[1])}

    # hierarchical clustering
    Z = linkage(topic_term_matrix, method='ward')
    dendro = dendrogram(Z, no_plot=True)

    num_topics = topic_term_matrix.shape[0]

    # --- Build per-node vectors and member lists (leaf ids 0..num_topics-1) ---
    cluster_vectors = {i: topic_term_matrix[i].astype(float) for i in range(num_topics)}
    cluster_members = {i: [i] for i in range(num_topics)}

    def top_words(vec, topn=6):
        idx = np.argsort(vec)[::-1][:topn]
        # vocabulary access: gensim Dictionary maps id->token; if not, try indexing
        words = []
        for i in idx:
            try:
                w = vocab[i]
            except Exception:
                try:
                    w = vocab[i]
                except Exception:
                    w = str(i)
            words.append(w)
        return words

    # Iterate merges to construct new clusters
    next_id = num_topics
    cluster_info = {}
    for k, (left, right, dist, sample_count) in enumerate(Z):
        left, right = int(left), int(right)
        new_vec = cluster_vectors[left] + cluster_vectors[right]
        cluster_vectors[next_id] = new_vec
        cluster_members[next_id] = cluster_members[left] + cluster_members[right]

        # top words for merged cluster
        cluster_info[next_id] = {
            "members": cluster_members[next_id],
            "top_words": ", ".join(top_words(new_vec, topn=6)),
            "distance": float(dist)
        }
        next_id += 1

    icoord = np.array(dendro['icoord'])
    dcoord = np.array(dendro['dcoord'])
    leaves = dendro['leaves']
    labels = [f"Topic {i}" for i in leaves]

    # build consistent topic colour map
    color_map = build_topic_color_map(num_topics)

    fig = go.Figure()

    # --- For each dendrogram segment, pick the representative cluster id and colour ---
    # SciPy produces one (icoord,dcoord) per merge in Z order. We'll map each merge k -> cid = num_topics + k
    for k in range(len(icoord)):
        cid = num_topics + k
        members = cluster_info.get(cid, {}).get("members", [])
        # decide representative topic:
        if members:
            # choose the single topic id among members with highest aggregate weight (or majority)
            # here we pick the modal member (most frequent leaf id) as representative
            counts = Counter(members)
            rep_topic = counts.most_common(1)[0][0]
            rep_color = color_map.get(f"Topic {rep_topic}", "#0D85D8")
        else:
            rep_color = "#0D85D8"

        fig.add_trace(go.Scatter(
            x=icoord[k],
            y=dcoord[k],
            mode='lines',
            line=dict(color=rep_color, width=2),
            hoverinfo='skip'
        ))

    # --- Add hoverable merge points (cluster summaries) with same rep colouring ---
    merge_x = []
    merge_y = []
    merge_text = []
    merge_colors = []

    max_y = float(np.max(dcoord)) if dcoord.size else 1.0
    min_y = float(np.min(dcoord)) if dcoord.size else 0.0
    span = max_y - min_y if (max_y - min_y) != 0 else 1.0

    for k, (xs, ys) in enumerate(zip(icoord, dcoord)):
        cid = num_topics + k
        cx = np.mean(xs[1:3])
        cy = ys[1]
        info = cluster_info.get(cid, {"members": [], "top_words": ""})
        members = info.get("members", [])
        member_topics = ", ".join([f"Topic {m}" for m in sorted(set(members))])
        topw = info.get("top_words", "")

        # representative colour (same rule as above)
        if members:
            counts = Counter(members)
            rep_topic = counts.most_common(1)[0][0]
            rep_color = color_map.get(f"Topic {rep_topic}", "#0D85D8")
        else:
            rep_color = "#0D85D8"

        merge_x.append(cx)
        merge_y.append(cy)
        merge_text.append(f"<b>Cluster {cid}</b><br>Topics: {member_topics}<br>Top words: {topw}")
        merge_colors.append(rep_color)

    fig.add_trace(go.Scatter(
        x=merge_x,
        y=merge_y,
        mode='markers',
        marker=dict(size=10, color=merge_colors, line=dict(color='black', width=0.5)),
        hovertemplate="%{text}<extra></extra>",
        text=merge_text,
        showlegend=False
    ))

    # --- Topic labels at the bottom, coloured by topic palette ---
    tick_x = [5 + 10 * i for i in range(len(labels))]
    label_y = min_y - span * 0.06

    tooltips = get_topic_tooltips(lda_model, topn=5)

    label_colors = []
    label_customdata = []
    for lbl in labels:
        tid = int(lbl.replace("Topic ", ""))
        label_colors.append(color_map.get(f"Topic {tid}", "#0D85D8"))
        label_customdata.append([tooltips.get(tid, "")])

    fig.add_trace(go.Scatter(
        x=tick_x,
        y=[label_y] * len(tick_x),
        mode='markers+text',
        marker=dict(size=0, color='rgba(0,0,0,0)'),
        text=labels,
        textfont=dict(color=label_colors),
        textposition='bottom center',
        customdata=label_customdata,
        hovertemplate="%{text}<br>Top words: %{customdata[0]}<extra></extra>",
        showlegend=False
    ))

    # Expand y axis to include labels and give bottom margin
    bottom_pad = span * 0.08
    top_pad = span * 0.05
    y_min = label_y - bottom_pad
    y_max = max_y + top_pad

    fig.update_layout(
        xaxis=dict(
            tickvals=tick_x,
            ticktext=[''] * len(labels),
            showgrid=False,
            zeroline=False,
            showticklabels=False
        ),
        yaxis=dict(range=[y_min, y_max], title='Distance'),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=0, r=0, t=0, b=0),
        font=dict(family=FONT_FAMILY, size=FONT_SIZE)
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

    tooltips = get_topic_tooltips(lda_model, topn=5)
    df["top_words"] = df["Topic"].apply(lambda t: tooltips[int(t.replace("Topic ", ""))])
    color_map = build_topic_color_map(lda_model.num_topics)
    fig = px.pie(df, names="Topic", values="Weight", height=420, hover_data=["top_words"], color_discrete_map=color_map)
    fig.update_layout(font=dict(family=FONT_FAMILY, size=FONT_SIZE))
    fig.update_traces(hovertemplate="<b>%{label}</b><br>Weight: %{value}<br>Top words: %{customdata[0]}<extra></extra>")
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    return plot_to_div(fig)


def create_topic_word_network(lda_model, topn=10):
    import networkx as nx
    import plotly.graph_objects as go

    G = nx.Graph()
    tooltips = get_topic_tooltips(lda_model, topn=5)  # short labels

    for topic_id in range(lda_model.num_topics):
        topic_label = f"Topic {topic_id}"
        G.add_node(topic_label, type='topic')
        for word, weight in lda_model.show_topic(topic_id, topn=topn):
            if not G.has_node(word):
                G.add_node(word, type='word')
            G.add_edge(topic_label, word, weight=weight)

    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'),
                            hoverinfo='none', mode='lines')

    node_x, node_y, node_text, node_color, node_custom = [], [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        color_map = build_topic_color_map(lda_model.num_topics)
        node_color.append(color_map.get(node, '#888888') if G.nodes[node]['type'] == 'topic' else '#888888')

        if G.nodes[node]['type'] == 'topic':
            # topic tooltip is short string e.g. "বাজার, অর্থনীতি, ব্যবসা"
            tid = int(node.replace('Topic ', ''))
            node_custom.append(tooltips.get(tid, ""))
        else:
            node_custom.append("")  # no tooltip for plain word nodes

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers+text', hoverinfo='text',
        marker=dict(size=12, color=node_color),
        text=node_text, textposition="bottom center",
        customdata=[[c] for c in node_custom],
        hovertemplate="%{text}<br>%{customdata[0]}<extra></extra>"
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(showlegend=False, hovermode='closest',
                                     margin=dict(b=0, l=0, r=0, t=0),
                                     paper_bgcolor='white', plot_bgcolor='white'))
    fig.update_layout(font=dict(family=FONT_FAMILY, size=FONT_SIZE))
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
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
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
