# -*- coding: utf-8 -*-
import logging
import os
import csv
import traceback
import re
from collections import Counter
from werkzeug.utils import secure_filename
from werkzeug.exceptions import HTTPException
from flask import Flask, request, render_template, send_file, send_from_directory, jsonify, flash, redirect, url_for
import nltk
from nltk.data import find
import io
import gensim
import gensim.corpora as corpora
import numpy as np
from preprocessing.pipeline import preprocess_documents
from analysis.corpus_stats import compute_corpus_stats
from lda.train import train_lda_model
from analysis.postprocess import compute_topic_semantics
from analysis.export import export_topics
from analysis.logs import finalize_training_log

from utils import (

    load_stopwords,
    convert_numpy_types
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
    create_corpus_top_tokens_bar
)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler("app.log", encoding="utf-8")]
)

# Flask app setup
app = Flask(__name__)
app.secret_key = os.environ.get('ANVAY_SECRET_KEY', 'anvay-dev-secret-change-in-production')
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

# Download NLTK resources if needed
def _ensure_nltk_resource(resource_path, download_name):
    """
    Download an NLTK resource if it is not already present.
    Catches both LookupError (resource missing) and OSError (path not found),
    since older NLTK versions raise OSError instead of LookupError for missing
    resources. The download itself is also wrapped so that unavailable packages
    (e.g. punkt_tab on NLTK < 3.9) fail silently rather than crashing startup.
    """
    try:
        find(resource_path)
    except (LookupError, OSError):
        try:
            nltk.download(download_name, quiet=True)
        except Exception:
            pass  # Non-fatal: package may not exist in this NLTK version

_ensure_nltk_resource('corpora/stopwords', 'stopwords')
_ensure_nltk_resource('tokenizers/punkt', 'punkt')
_ensure_nltk_resource('tokenizers/punkt_tab', 'punkt_tab')  # NLTK 3.9+ only; silently skipped on older versions
_ensure_nltk_resource('corpora/wordnet', 'wordnet')


def sanitize_filename(filename):
    """
    Unicode-safe filename sanitiser. Preserves Bengali (and all Unicode) filenames.

    The original re.sub(r'[^\\w...]') approach strips Unicode combining marks
    (category Mn) such as Bengali vowel signs/matras (া ি ী ু ূ ে ো etc.),
    hasanta (্), and anusvara (ং). This causes Bengali filenames to appear as bare
    consonant skeletons in graphs and reports. Fix: use a denylist of genuinely
    dangerous filename characters instead of an allowlist.
    """
    # Take basename only — prevents path traversal
    name = os.path.basename(filename)
    # Strip only characters that are genuinely dangerous in filenames:
    # null bytes, path separators, and shell metacharacters.
    # All Unicode letters, combining marks (vowel signs, matras), digits are kept.
    name = re.sub(r'[\x00/\\:*?"<>|]', '_', name)
    # Collapse runs of underscores/spaces
    name = re.sub(r'[\s_]+', '_', name).strip('_')
    return name or 'upload'


def process_txt_files(file_paths, config, custom_stopwords=None):
    """
    Run preprocessing → LDA training → postprocessing.

    Parameters
    ----------
    file_paths : list[str]
    config     : AnalysisConfig
    custom_stopwords : set | None
    """
    
    # Capture gensim training logs for display

    log_stream = io.StringIO()
    handler = logging.StreamHandler(log_stream)
    handler.setLevel(logging.INFO)
    gensim_logger = logging.getLogger('gensim')
    gensim_logger.addHandler(handler)

    all_tokens, raw_texts, doc_names = preprocess_documents(
        file_paths=file_paths,
        remove_stopwords=config.remove_stopwords,
        custom_stopwords=custom_stopwords,
        normalisation=config.normalisation,
        normalisation_order=config.normalisation_order,
        percent=config.percent_most_common,
        ngram=config.ngram,
        language=config.language
    )


    id2word = corpora.Dictionary(all_tokens)
    id2word.filter_extremes(no_below=config.no_below, no_above=config.no_above, keep_n=None)
    if len(id2word) == 0:
        raise ValueError("Dictionary is empty after filtering. Adjust no_below/no_above.")
    corpus = [id2word.doc2bow(tok) for tok in all_tokens]

    lda_model = train_lda_model(
        corpus=corpus,
        id2word=id2word,
        num_topics=config.num_topics,
        iterations=config.iterations,
        passes=config.passes,
        chunk_size=config.chunk_size,
        alpha=config.alpha,
        eta=config.eta,
        per_word_topics=config.per_word_topics,
        minimum_probability=config.minimum_probability,
        use_multicore=config.use_multicore,
        log_stream=log_stream
    )


    overview_stats, top_tokens, top_token_text = compute_corpus_stats(
        all_tokens,
        config.normalisation_order
    )


    # Semantic postprocessing
    relevance_topics, topic_labels, representative_sents = compute_topic_semantics(
        lda_model=lda_model,
        corpus=corpus,
        id2word=id2word,
        raw_texts=raw_texts,
        doc_names=doc_names,
        language=config.language
    )

    # Export topics
    txt_path, csv_path = export_topics(
        lda_model=lda_model,
        num_topics=config.num_topics,
        result_folder=RESULT_FOLDER
    )

    # Finalise logs
    training_log = finalize_training_log(
        gensim_logger,
        handler,
        log_stream
    )

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
        safe_name = sanitize_filename(f.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], safe_name)
        f.save(path)
        file_paths.append(path)

    # Hyperparameters
    from config.analysis_config import build_analysis_config

    config = build_analysis_config(request.form)

 

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
        ) = process_txt_files(file_paths, config, custom_stopwords_set)

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
        for i in range(config.num_topics)
    }
    doc_topic_matrix = {
        doc_names[i]: [
            dict(lda_model.get_document_topics(corpus[i])).get(tid, 0.0)
            for tid in range(config.num_topics)
        ]
        for i in range(len(corpus))
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
        writer.writerow(['Document'] + [f'Topic {i}' for i in range(config.num_topics)])
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
            num_topics=config.num_topics
       

        )
    except Exception as e:
        logging.error("Template rendering failed:\n" + traceback.format_exc())
        return f"Template rendering error: {e}", 500


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

