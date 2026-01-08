# -*- coding: utf-8 -*-
import logging
import os
import csv
import traceback
from collections import Counter
from werkzeug.utils import secure_filename
from werkzeug.exceptions import HTTPException
from flask import Flask, request, render_template, send_file, send_from_directory, jsonify, flash, redirect, url_for
from contextlib import redirect_stdout
import nltk
from nltk.corpus import stopwords
from nltk.data import find
import io
import gensim
import gensim.corpora as corpora
import numpy as np

from utils import (
    remove_most_frequent_words,
    custom_bengali_tokenize,
    stem_tokens,
    load_list_from_file,
    load_stopwords,
    parse_hyperparam,
    convert_numpy_types,
    get_relevance_weighted_words,
    generate_topic_labels,
    remove_punctuation
)
from viz import (
    create_interactive_scatter,
    create_interactive_bar_charts,
    create_interactive_heatmap,
    create_interactive_topic_evolution,
    create_interactive_clustering,
    create_interactive_topic_distribution,
    create_topic_prevalence_pie,
    create_topic_word_network,
    prepare_topic_doc_drilldown,
    create_corpus_top_tokens_bar,
    get_representative_sentences_custom
)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler("app.log", encoding="utf-8")]
)

# Flask app setup
app = Flask(__name__)
app.secret_key = 'anvay-secret-key'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB

# Directories
BASE_DIR = os.getcwd()
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
RESULT_FOLDER = os.path.join(BASE_DIR, 'results')
STOPWORDS_FOLDER = os.path.join(BASE_DIR, 'stopwords')
STATIC_FOLDERS = {
    'plotly': os.path.join(RESULT_FOLDER, 'plotly'),
    'bokeh': os.path.join(RESULT_FOLDER, 'bokeh'),
    'seaborn': os.path.join(RESULT_FOLDER, 'seaborn'),
    'html': os.path.join(RESULT_FOLDER, 'html')
}

# Ensure directories exist

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(STOPWORDS_FOLDER, exist_ok=True)
for path in STATIC_FOLDERS.values():
    os.makedirs(path, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['STOPWORDS_FOLDER'] = STOPWORDS_FOLDER

# Load stemmer roots/suffixes once
NOUN_ROOTS = set(load_list_from_file(os.path.join(STOPWORDS_FOLDER, 'noun_roots.txt')))
VERB_ROOTS = set(load_list_from_file(os.path.join(STOPWORDS_FOLDER, 'verb_roots.txt')))
NOUN_SUFFIXES = load_list_from_file(os.path.join(STOPWORDS_FOLDER, 'noun_suffixes.txt'))
VERB_SUFFIXES = load_list_from_file(os.path.join(STOPWORDS_FOLDER, 'verb_suffixes.txt'))


# Download NLTK resources if needed
try:
    find('corpora/stopwords')
    find('tokenizers/punkt')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')


def process_txt_files(
    file_paths, num_topics, iterations, passes, minimum_probability,
    chunk_size, ngram, alpha, eta, per_word_topics,
    no_below, no_above, use_stemming, use_multicore,
    percent, remove_stopwords, custom_stopwords=None
):
    
    # Capture gensim training logs for display

    log_stream = io.StringIO()
    handler = logging.StreamHandler(log_stream)
    handler.setLevel(logging.INFO)
    gensim_logger = logging.getLogger('gensim')
    gensim_logger.addHandler(handler)

    # Build stopwords set
    stop_words = set(stopwords.words('bengali')) if remove_stopwords else set()
    if custom_stopwords:
        stop_words.update(custom_stopwords)
    

    all_tokens, raw_texts, doc_names = [], [], []
    for path in file_paths:
        with open(path, 'r', encoding='utf-8') as f:
            raw = f.read()
            clean_text = remove_punctuation(raw)

        tokens = custom_bengali_tokenize(clean_text)
        tokens = [t for t in tokens if len(t) > 1 and t not in stop_words]
        if use_stemming:
            tokens = stem_tokens(
                [tokens], NOUN_ROOTS, VERB_ROOTS, NOUN_SUFFIXES, VERB_SUFFIXES
            )[0]
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

    # Create dictionary and corpus
    id2word = corpora.Dictionary(all_tokens)
    id2word.filter_extremes(no_below=no_below, no_above=no_above, keep_n=None)
    if len(id2word) == 0:
        raise ValueError("Dictionary is empty after filtering. Adjust no_below/no_above.")
    corpus = [id2word.doc2bow(tok) for tok in all_tokens]
    with redirect_stdout(log_stream):
    # Train LDA
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

    # Relevance-weighted top words & labels
    relevance_topics = get_relevance_weighted_words(lda_model, corpus, id2word, lambda_val=0.6)
    topic_labels = generate_topic_labels(relevance_topics)

    # Representative sentences
    representative_sents = get_representative_sentences_custom(
        lda_model, corpus, raw_texts, id2word, doc_names=doc_names
    )

    # Save topics to TXT & CSV
    topics = lda_model.show_topics(num_topics=num_topics, formatted=False)
    txt_path = os.path.join(RESULT_FOLDER, 'topics.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        for t in topics:
            f.write(f"Topic {t[0]}:\n{', '.join(w[0] for w in t[1])}\n\n")

    csv_path = os.path.join(RESULT_FOLDER, 'topics.csv')
    with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Topic', 'Words'])
        for t in topics:
            writer.writerow([t[0], ", ".join(w[0] for w in t[1])])

    # Corpus & token stats
    total_docs = len(all_tokens)
    flat_tokens = [t for doc in all_tokens for t in doc]
    total_tokens = len(flat_tokens)
    vocab_size = len(set(flat_tokens))
    avg_len = round(np.mean([len(doc) for doc in all_tokens]), 2)
    top_tokens = Counter(flat_tokens).most_common(10)
    top_token_text = [
        {"token": token, "percent": round(count / total_tokens * 100, 2)}
        for token, count in top_tokens
    ]
    overview_stats = {
        "Total Documents": total_docs,
        "Total Tokens": total_tokens,
        "Vocabulary Size": vocab_size,
        "Average Document Length": avg_len,
        "Shortest Document Length": min(len(doc) for doc in all_tokens),
        "Longest Document Length": max(len(doc) for doc in all_tokens),
    }

    gensim_logger.removeHandler(handler)
    full_log = log_stream.getvalue()

    training_log = extract_observational_lines(full_log)

    return (
        txt_path,
        csv_path,
        lda_model,
        corpus,
        id2word,
        all_tokens,
        overview_stats,
        top_tokens,
        top_token_text,
        raw_texts,
        doc_names,
        relevance_topics,
        topic_labels,
        representative_sents,
        training_log
        
    )


def save_plot_html(html_str, filename, folder_key):
    folder = STATIC_FOLDERS.get(folder_key)
    if not folder:
        logging.warning(f"No folder found for key: {folder_key}")
        return None
    path = os.path.join(folder, filename)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html_str)
    return filename


@app.route('/')
def upload_file():
    return render_template('upload.html')


@app.route('/process', methods=['POST'])
def process_files():
    files = request.files.getlist('files[]')
    if not files or files[0].filename == '':
        flash("No files selected.", "danger")
        return redirect(url_for('upload_file'))
    if not all(f.filename.endswith('.txt') for f in files):
        flash("Only .txt files are supported.", "danger")
        return redirect(url_for('upload_file'))

    file_paths = []
    
    for f in files:
        try:
        # sanitize the original filename (keeping Unicode characters)
            safe_name = secure_filename(f.filename, allow_unicode=True)
        except TypeError:
            safe_name = os.path.basename(f.filename).replace(os.sep, "_")
        # build the absolute path
        path = os.path.join(app.config['UPLOAD_FOLDER'], safe_name)
        # save to disk
        f.save(path)
        file_paths.append(path) 

    # Hyperparameters
    alpha = parse_hyperparam(request.form.get("alpha", "symmetric"))
    eta = parse_hyperparam(request.form.get("eta", "symmetric"))
    num_topics = int(request.form.get('num_topics', 10))
    iterations = int(request.form.get('iterations', 40))
    passes = int(request.form.get('passes', 10))
    chunk_size = int(request.form.get('chunk_size', 200))
    ngram = request.form.get('ngram', 'unigram')
    per_word_topics = request.form.get('per_word_topics', 'false').lower() == 'true'
    no_below = int(request.form.get('no_below', 1))
    no_above = float(request.form.get('no_above', 0.9))
    use_stemming = request.form.get('use_stemming', '') == 'rule_based'
    use_multicore = request.form.get('use_multicore', 'false').lower() == 'true'
    percent = float(request.form.get('percent_most_common', 0))
    minimum_probability = 0.1
 

    # Load custom stopwords if provided
    custom_stopwords_set = set()
    if 'custom_stopwords' in request.files:
        custom = request.files['custom_stopwords']
        if custom and custom.filename.strip():
            sw_path = os.path.join(
                app.config['STOPWORDS_FOLDER'], secure_filename(custom.filename)
            )
            os.makedirs(app.config['STOPWORDS_FOLDER'], exist_ok=True)
            custom.save(sw_path)
            custom_stopwords_set = load_stopwords(sw_path)

    # Run processing/train
    try:
        (
            txt_path,
            csv_path,
            lda_model,
            corpus,
            id2word,
            all_tokens,
            overview_stats,
            top_tokens,
            top_token_text,
            raw_texts,
            doc_names,
            relevance_topics,
            topic_labels,
            representative_sents,
            training_log
       
        ) = process_txt_files(
            file_paths,
            num_topics,
            iterations,
            passes,
            minimum_probability,
            chunk_size,
            ngram,
            alpha,
            eta,
            per_word_topics,
            no_below,
            no_above,
            use_stemming,
            use_multicore,
            percent,
            True,
            custom_stopwords_set
        )
    except ValueError as ve:
        app.logger.warning(f"User-level processing error: {ve}")
        return render_template("error.html", error={
            'code': 'Invalid Parameters',
            'name': 'Topic Model Failed',
            'description': str(ve)
        }), 400
    except Exception as e:
        app.logger.error("Unhandled error during processing:\n" + traceback.format_exc())
        return render_template("error.html", error={
            'code': 500,
            'name': 'Internal Server Error',
            'description': 'An unexpected error occurred while processing your text. Please try again or adjust your settings.'
        }), 500
    

    # Save the "Top Tokens" bar
    top_tokens_bar = create_corpus_top_tokens_bar(top_tokens)
    save_plot_html(top_tokens_bar, 'top_tokens_bar.html', 'plotly')

    # Skip heavy coherence computation for now
    coherence_score = None

    # Prepare drilldown paragraphs
    topic_doc_data = prepare_topic_doc_drilldown(
        lda_model, corpus, doc_names=doc_names, raw_texts=raw_texts, min_weight=0.2
    )
    clean_topic_doc_data = convert_numpy_types(topic_doc_data)

    # Topic highlights (sentences)
    topic_highlights = None  # or call generate_topic_highlights if you prefer

    topic_words = {
    i: [word for word, _ in lda_model.show_topic(i, topn=10)]
    for i in range(num_topics)
    }
    doc_topic_matrix = {
    doc_names[i]: [
        dict(lda_model.get_document_topics(corpus[i])).get(tid, 0.0)
        for tid in range(num_topics)
    ]
    for i in range(min(25, len(corpus)))  # Limit for display
    }

    # Save topic_words to CSV and TXT
    topic_words_csv_path = os.path.join(app.config['RESULT_FOLDER'], 'topic_words.csv')
    with open(topic_words_csv_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Topic', 'Top Words'])
        for tid, words in topic_words.items():
            writer.writerow([tid, ", ".join(words)])

    topic_words_txt_path = os.path.join(app.config['RESULT_FOLDER'], 'topic_words.txt')
    with open(topic_words_txt_path, 'w', encoding='utf-8') as f:
        for tid, words in topic_words.items():
            f.write(f"Topic {tid}: {', '.join(words)}\n")

# Save doc_topic_matrix to CSV and TXT
    doc_weights_csv_path = os.path.join(app.config['RESULT_FOLDER'], 'doc_topic_weights.csv')
    with open(doc_weights_csv_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Document'] + [f'Topic {i}' for i in range(num_topics)])
        for doc, weights in doc_topic_matrix.items():
            writer.writerow([doc] + [round(w, 4) for w in weights])

    doc_weights_txt_path = os.path.join(app.config['RESULT_FOLDER'], 'doc_topic_weights.txt')
    with open(doc_weights_txt_path, 'w', encoding='utf-8') as f:
        for doc, weights in doc_topic_matrix.items():
            line = f"{doc}: " + ", ".join(f"Topic {i}: {round(w, 4)}" for i, w in enumerate(weights)) + "\n"
            f.write(line)


    # Build all visualizations
    visualizations = {
        'topics_txt': ('Download TXT', 'topics.txt', 'results'),
        'topics_csv': ('Download CSV', 'topics.csv', 'results'),
        'scatter': ('Scatter Plot', save_plot_html(create_interactive_scatter(lda_model, corpus), 'scatter.html', 'bokeh'), 'bokeh'),
        'bars': ('Bar Chart', save_plot_html(create_interactive_bar_charts(lda_model), 'bars.html', 'bokeh'), 'bokeh'),
        'heatmap': ('Heatmap', save_plot_html(create_interactive_heatmap(lda_model), 'heatmap.html', 'seaborn'), 'seaborn'),
        'evolution': ('Topic Evolution', save_plot_html(create_interactive_topic_evolution(lda_model, corpus, doc_names=doc_names), 'evolution.html', 'seaborn'), 'seaborn'),
        'clustering': ('Clustering', save_plot_html(create_interactive_clustering(lda_model), 'clustering.html', 'seaborn'), 'seaborn'),
        'distribution': ('Distribution', save_plot_html(create_interactive_topic_distribution(lda_model, corpus, doc_names=doc_names), 'distribution.html', 'seaborn'), 'seaborn'),
        'prevalence_pie': ('Topic Prevalence Pie', save_plot_html(create_topic_prevalence_pie(lda_model, corpus), 'prevalence_pie.html', 'plotly'), 'plotly'),
        'word_network': ('Topic Word Graph', save_plot_html(create_topic_word_network(lda_model), 'word_network.html', 'plotly'), 'plotly'),
        'overview': ('Training Overview', 'top_tokens_bar.html', 'plotly')
    }

 

    try:
        return render_template(
            'result.html',
            coherence_score=coherence_score,
            visualizations=visualizations,
            topic_doc_data=clean_topic_doc_data,
            overview_stats=overview_stats,
            topic_highlights=topic_highlights,
            top_token_text=top_token_text,
            relevance_topics=relevance_topics,
            topic_labels=topic_labels,
            representative_sents=representative_sents,
            doc_names=doc_names,
            training_log=training_log,
            topic_words=topic_words,
            doc_topic_matrix=doc_topic_matrix,
            num_topics=num_topics
       

        )
    except Exception as e:
        logging.error("Template rendering failed:\n" + traceback.format_exc())
        return f"Template rendering error: {e}", 500

def capture_lda_training_log():
    buffer = io.StringIO()
    handler = logging.StreamHandler(buffer)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)

    lda_logger = logging.getLogger('gensim')
    lda_logger.setLevel(logging.INFO)
    lda_logger.addHandler(handler)

    return buffer, handler, lda_logger

def extract_observational_lines(full_log):
    lines = full_log.splitlines()
    parsed = {
        "token_filtering": [],
        "config": [],
        "progress": [],
        "convergence": [],
        "perplexity": [],
        "final": []
    }

    for line in lines:
        if "built Dictionary<" in line or "discarding" in line or "keeping" in line or "resulting dictionary" in line:
            parsed["token_filtering"].append(line)
        elif "using symmetric" in line or "running online LDA training" in line or "training LDA model" in line:
            parsed["config"].append(line)
        elif line.startswith("PROGRESS:"):
            parsed["progress"].append(line)
        elif "documents converged" in line:
            parsed["convergence"].append(line)
        elif "perplexity estimate" in line:
            parsed["perplexity"].append(line)
        elif "LdaMulticore lifecycle event" in line:
            parsed["final"].append(line)

    return parsed

@app.route('/results/<folder>/<filename>')
def view_result_file(folder, filename):
    folder_path = STATIC_FOLDERS.get(folder)
    if folder_path and os.path.exists(os.path.join(folder_path, filename)):
        return send_from_directory(folder_path, filename)
    return "File not found", 404


@app.route('/download_file/<filename>')
def download_file(filename):
    path = os.path.join(RESULT_FOLDER, filename)
    if not os.path.exists(path):
        return "File not found", 404
    return send_file(path, as_attachment=True)


@app.route('/filter')
def filter_topics():
    query = request.args.get('query', '').lower()
    results = []
    try:
        with open(os.path.join(RESULT_FOLDER, 'topics.txt'), 'r', encoding='utf-8') as f:
            current = None
            for line in f:
                if line.startswith('Topic'):
                    current = line.strip()
                elif query in line.lower():
                    results.append({'topic': current, 'keywords': line.strip()})
    except FileNotFoundError:
        return jsonify([])
    return jsonify(results)


# Static pages
@app.route('/about')
def about():    return render_template('about.html')
@app.route('/docs')
def docs():     return render_template('docs.html')
@app.route('/contact')
def contact():  return render_template('contact.html')
@app.route('/sitemap')
def sitemap():  return render_template('sitemap.html')
@app.route('/privacy')
def privacy():  return render_template('privacy.html')
@app.route('/terms')
def terms():    return render_template('terms.html')


@app.errorhandler(404)
def handle_404(e):
    return render_template("error.html", error={
        'code': 404,
        'name': 'Page Not Found',
        'description': 'The page you requested could not be found.'
    }), 404

@app.errorhandler(500)
def handle_500(e):
    return render_template("error.html", error={
        'code': 500,
        'name': 'Internal Server Error',
        'description': 'Something went wrong on our end. Please try again later.'
    }), 500

@app.errorhandler(503)
def handle_503(e):
    return render_template("error.html", error={
        'code': 503,
        'name': 'Server Busy',
        'description': 'The server is temporarily unavailable or overloaded. Please try again shortly.'
    }), 503


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)

