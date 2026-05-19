# -*- coding: utf-8 -*-
"""
app.py — anvay (production branch)

Security hardening applied over main:
  - Per-request job isolation (UUID-scoped upload + result directories)
  - Input validation (file content, size, count, hyperparameter bounds)
  - Upload cleanup in try/finally (uploads deleted after processing)
  - Result cleanup on each new request (dirs older than 2 hours swept)
  - Rate limiting via Flask-Limiter
  - Hardened download/result routes (path traversal protection)
  - Security event logging
  - DEBUG forced off; secret key required from environment
"""

import logging
import os
import csv
import shutil
import traceback
import re
import uuid
from collections import Counter
from werkzeug.utils import secure_filename
from werkzeug.exceptions import HTTPException
from flask import Flask, request, render_template, send_file, send_from_directory, jsonify, flash, redirect, url_for, abort
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
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

from utils import load_stopwords, convert_numpy_types
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
)
from security import (
    is_valid_uuid,
    validate_upload,
    validate_file_count,
    validate_hyperparams,
    safe_result_path,
    cleanup_old_jobs,
    get_security_logger,
    ALLOWED_VIZ_FOLDERS,
    ALLOWED_DOWNLOAD_FILES,
    MAX_FILES,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
os.makedirs('logs', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,                          # INFO in production, not DEBUG
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/app.log", encoding="utf-8"),
    ],
)

# Separate security log — easier to grep and ship to a SIEM later.
sec_log = get_security_logger()
sec_handler = logging.FileHandler("logs/security.log", encoding="utf-8")
sec_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
sec_log.addHandler(sec_handler)
sec_log.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)

# Secret key MUST come from the environment in production.
# If the variable is missing the app refuses to start — this is intentional.
_secret = os.environ.get('ANVAY_SECRET_KEY')
if not _secret:
    raise RuntimeError(
        "ANVAY_SECRET_KEY environment variable is not set. "
        "Set it to a long random string before starting the server."
    )
app.secret_key = _secret

# Never run with debug on in production.
app.config['DEBUG'] = False

# Total request body limit (all files combined).
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5 MB

# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------
# flask-limiter >= 3.x API. Uses in-process memory store — adequate for a
# single-worker deployment. Switch to Redis storage_uri for multi-worker.

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["300 per day", "60 per hour"],
    storage_uri="memory://",
)

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER    = os.path.join(BASE_DIR, 'uploads')
RESULT_FOLDER    = os.path.join(BASE_DIR, 'results')
STOPWORDS_FOLDER = os.path.join(BASE_DIR, 'stopwords')
LOG_DIR          = os.path.join(BASE_DIR, 'logs')

for d in (UPLOAD_FOLDER, RESULT_FOLDER, STOPWORDS_FOLDER, LOG_DIR):
    os.makedirs(d, exist_ok=True)

app.config['UPLOAD_FOLDER']    = UPLOAD_FOLDER
app.config['RESULT_FOLDER']    = RESULT_FOLDER
app.config['STOPWORDS_FOLDER'] = STOPWORDS_FOLDER

# ---------------------------------------------------------------------------
# NLTK resources
# ---------------------------------------------------------------------------

def _ensure_nltk_resource(resource_path, download_name):
    try:
        find(resource_path)
    except (LookupError, OSError):
        try:
            nltk.download(download_name, quiet=True)
        except Exception:
            pass

_ensure_nltk_resource('corpora/stopwords',    'stopwords')
_ensure_nltk_resource('tokenizers/punkt',     'punkt')
_ensure_nltk_resource('tokenizers/punkt_tab', 'punkt_tab')
_ensure_nltk_resource('corpora/wordnet',      'wordnet')

# ---------------------------------------------------------------------------
# Filename sanitiser
# ---------------------------------------------------------------------------

def sanitize_filename(filename: str) -> str:
    """
    Unicode-safe filename sanitiser. Preserves Bengali (and all Unicode)
    filenames while stripping genuinely dangerous characters.
    """
    name = os.path.basename(filename)
    name = re.sub(r'[\x00/\\:*?"<>|]', '_', name)
    name = re.sub(r'[\s_]+', '_', name).strip('_')
    return name or 'upload'

# ---------------------------------------------------------------------------
# Plot saving (job-scoped)
# ---------------------------------------------------------------------------

def save_plot_html(html_str: str, filename: str, subfolder: str, job_result_dir: str) -> str:
    """
    Write a plot HTML string to job_result_dir/subfolder/filename.
    Returns filename (used as the key in the visualizations dict).
    """
    folder = os.path.join(job_result_dir, subfolder)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html_str)
    return filename

# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def process_txt_files(file_paths, config, job_result_dir, custom_stopwords=None):
    """
    Run preprocessing → LDA training → postprocessing.

    Parameters
    ----------
    file_paths      : list[str]
    config          : AnalysisConfig
    job_result_dir  : str  — job-scoped result directory
    custom_stopwords: set | None
    """

    log_stream = io.StringIO()
    handler    = logging.StreamHandler(log_stream)
    handler.setLevel(logging.INFO)
    gensim_logger = logging.getLogger('gensim')
    gensim_logger.addHandler(handler)

    all_tokens, raw_texts, doc_names = preprocess_documents(
        file_paths        = file_paths,
        remove_stopwords  = config.remove_stopwords,
        custom_stopwords  = custom_stopwords,
        normalisation     = config.normalisation,
        normalisation_order = config.normalisation_order,
        percent           = config.percent_most_common,
        ngram             = config.ngram,
        language          = config.language,
    )

    id2word = corpora.Dictionary(all_tokens)
    id2word.filter_extremes(
        no_below = config.no_below,
        no_above = config.no_above,
        keep_n   = None,
    )
    if len(id2word) == 0:
        raise ValueError(
            "Dictionary is empty after filtering. "
            "Try lowering no_below or raising no_above."
        )
    corpus = [id2word.doc2bow(tok) for tok in all_tokens]

    lda_model = train_lda_model(
        corpus             = corpus,
        id2word            = id2word,
        num_topics         = config.num_topics,
        iterations         = config.iterations,
        passes             = config.passes,
        chunk_size         = config.chunk_size,
        alpha              = config.alpha,
        eta                = config.eta,
        per_word_topics    = config.per_word_topics,
        minimum_probability= config.minimum_probability,
        use_multicore      = config.use_multicore,
        log_stream         = log_stream,
    )

    overview_stats, top_tokens, top_token_text = compute_corpus_stats(
        all_tokens,
        config.normalisation_order,
    )

    relevance_topics, topic_labels, representative_sents = compute_topic_semantics(
        lda_model  = lda_model,
        corpus     = corpus,
        id2word    = id2word,
        raw_texts  = raw_texts,
        doc_names  = doc_names,
        language   = config.language,
    )

    # Write topics.txt / topics.csv into the job-scoped result dir.
    txt_path, csv_path = export_topics(
        lda_model     = lda_model,
        num_topics    = config.num_topics,
        result_folder = job_result_dir,        # ← job-scoped
    )

    training_log = finalize_training_log(gensim_logger, handler, log_stream)

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
        training_log,
    )

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/')
def upload_file():
    return render_template('upload.html')


@app.route('/process', methods=['POST'])
@limiter.limit("6 per minute; 30 per hour")
def process_files():
    """
    Main processing endpoint.

    Security steps (in order):
      1. Sweep old result directories (2 h TTL).
      2. Validate file count.
      3. Validate each file: extension, size, content.
      4. Validate numeric hyperparameters.
      5. Assign a UUID job_id; create per-job upload + result dirs.
      6. Save uploads to job dir.
      7. Run processing.
      8. Delete upload dir in finally block regardless of outcome.
    """

    # 1. Clean up stale result dirs from previous jobs.
    cleanup_old_jobs(RESULT_FOLDER)

    ip = request.remote_addr

    # ------------------------------------------------------------------
    # 2. File count
    # ------------------------------------------------------------------
    files = request.files.getlist('files[]')
    if not files or files[0].filename == '':
        flash("No files selected.", "danger")
        return redirect(url_for('upload_file'))

    ok, err = validate_file_count(files)
    if not ok:
        sec_log.warning(f"FILE_COUNT_EXCEEDED ip={ip} count={len(files)}")
        flash(err, "danger")
        return redirect(url_for('upload_file'))

    # ------------------------------------------------------------------
    # 3. Per-file validation
    # ------------------------------------------------------------------
    for f in files:
        ok, err = validate_upload(f, f.filename)
        if not ok:
            sec_log.warning(f"INVALID_UPLOAD ip={ip} filename={f.filename!r} reason={err}")
            flash(err, "danger")
            return redirect(url_for('upload_file'))

    # ------------------------------------------------------------------
    # 4. Hyperparameter validation
    # ------------------------------------------------------------------
    param_errors = validate_hyperparams(request.form)
    if param_errors:
        for e in param_errors:
            flash(e, "danger")
        return redirect(url_for('upload_file'))

    # ------------------------------------------------------------------
    # 5. Assign job ID and create per-job directories
    # ------------------------------------------------------------------
    job_id = str(uuid.uuid4())
    job_upload_dir = os.path.join(UPLOAD_FOLDER, job_id)
    job_result_dir = os.path.join(RESULT_FOLDER, job_id)

    for sub in ('', 'plotly', 'bokeh', 'seaborn', 'html'):
        os.makedirs(os.path.join(job_result_dir, sub), exist_ok=True)
    os.makedirs(job_upload_dir, exist_ok=True)

    sec_log.info(f"JOB_START ip={ip} job={job_id} files={len(files)}")

    file_paths = []

    try:
        # --------------------------------------------------------------
        # 6. Save uploads
        # --------------------------------------------------------------
        for f in files:
            safe_name = sanitize_filename(f.filename)
            path = os.path.join(job_upload_dir, safe_name)
            f.save(path)
            file_paths.append(path)

        # Load custom stopwords (optional)
        custom_stopwords_set = set()
        if 'custom_stopwords' in request.files:
            custom = request.files['custom_stopwords']
            if custom and custom.filename.strip():
                sw_name = secure_filename(custom.filename)
                sw_path = os.path.join(job_upload_dir, sw_name)
                custom.save(sw_path)
                custom_stopwords_set = load_stopwords(sw_path)

        # Build config
        from config.analysis_config import build_analysis_config
        config = build_analysis_config(request.form)

        # --------------------------------------------------------------
        # 7. Run processing
        # --------------------------------------------------------------
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
                training_log,
            ) = process_txt_files(file_paths, config, job_result_dir, custom_stopwords_set)

        except ValueError as ve:
            sec_log.info(f"JOB_PARAM_ERROR ip={ip} job={job_id} err={ve}")
            app.logger.warning(f"User-level processing error: {ve}")
            return render_template("error.html", error={
                'code': 'Invalid Parameters',
                'name': 'Topic Model Failed',
                'description': str(ve),
            }), 400

        except Exception:
            sec_log.error(f"JOB_CRASH ip={ip} job={job_id}")
            app.logger.error("Unhandled error during processing:\n" + traceback.format_exc())
            return render_template("error.html", error={
                'code': 500,
                'name': 'Internal Server Error',
                'description': (
                    'An unexpected error occurred while processing your text. '
                    'Please try again or adjust your settings.'
                ),
            }), 500

        # --------------------------------------------------------------
        # Helper closure — keeps save_plot_html calls tidy below
        # --------------------------------------------------------------
        def _save(html, fname, sub):
            return save_plot_html(html, fname, sub, job_result_dir)

        # Training overview bar
        _save(create_corpus_top_tokens_bar(top_tokens), 'top_tokens_bar.html', 'plotly')

        coherence_score = None

        topic_doc_data      = prepare_topic_doc_drilldown(
            lda_model, corpus, doc_names=doc_names, raw_texts=raw_texts, min_weight=0.2
        )
        clean_topic_doc_data = convert_numpy_types(topic_doc_data)
        topic_highlights    = None

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

        # Write topic CSVs / TXTs into the job result dir.
        topic_words_csv = os.path.join(job_result_dir, 'topic_words.csv')
        with open(topic_words_csv, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Topic', 'Top Words'])
            for tid, words in topic_words.items():
                writer.writerow([tid, ", ".join(words)])

        topic_words_txt = os.path.join(job_result_dir, 'topic_words.txt')
        with open(topic_words_txt, 'w', encoding='utf-8') as f:
            for tid, words in topic_words.items():
                f.write(f"Topic {tid}: {', '.join(words)}\n")

        doc_weights_csv = os.path.join(job_result_dir, 'doc_topic_weights.csv')
        with open(doc_weights_csv, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Document'] + [f'Topic {i}' for i in range(config.num_topics)])
            for doc, weights in doc_topic_matrix.items():
                writer.writerow([doc] + [round(w, 4) for w in weights])

        doc_weights_txt = os.path.join(job_result_dir, 'doc_topic_weights.txt')
        with open(doc_weights_txt, 'w', encoding='utf-8') as f:
            for doc, weights in doc_topic_matrix.items():
                line = (
                    f"{doc}: "
                    + ", ".join(f"Topic {i}: {round(w, 4)}" for i, w in enumerate(weights))
                    + "\n"
                )
                f.write(line)

        # Build visualizations
        _save(create_interactive_scatter(lda_model, corpus),                           'scatter.html',      'bokeh')
        _save(create_interactive_bar_charts(lda_model),                                'bars.html',         'bokeh')
        _save(create_interactive_heatmap(lda_model),                                   'heatmap.html',      'seaborn')
        _save(create_interactive_topic_evolution(lda_model, corpus, doc_names=doc_names), 'evolution.html', 'seaborn')
        _save(create_interactive_clustering(lda_model),                                'clustering.html',   'seaborn')
        _save(create_interactive_topic_distribution(lda_model, corpus, doc_names=doc_names), 'distribution.html', 'seaborn')
        _save(create_topic_prevalence_pie(lda_model, corpus),                          'prevalence_pie.html', 'plotly')
        _save(create_topic_word_network(lda_model),                                    'word_network.html', 'plotly')

        sec_log.info(f"JOB_DONE ip={ip} job={job_id}")

        try:
            return render_template(
                'result.html',
                job_id               = job_id,
                coherence_score      = coherence_score,
                topic_doc_data       = clean_topic_doc_data,
                overview_stats       = overview_stats,
                topic_highlights     = topic_highlights,
                top_token_text       = top_token_text,
                relevance_topics     = relevance_topics,
                topic_labels         = topic_labels,
                representative_sents = representative_sents,
                doc_names            = doc_names,
                training_log         = training_log,
                topic_words          = topic_words,
                doc_topic_matrix     = doc_topic_matrix,
                num_topics           = config.num_topics,
            )
        except Exception:
            logging.error("Template rendering failed:\n" + traceback.format_exc())
            return "Template rendering error", 500

    finally:
        # ------------------------------------------------------------------
        # 8. Always delete the upload directory — win or lose.
        # Result directory is kept for 2 h (served to user; swept next request).
        # ------------------------------------------------------------------
        shutil.rmtree(job_upload_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Result file serving (job-scoped, hardened)
# ---------------------------------------------------------------------------

@app.route('/results/<job_id>/<folder>/<filename>')
def view_result_file(job_id, folder, filename):
    """Serve a visualisation file. Validates job_id, folder, and filename."""

    if not is_valid_uuid(job_id):
        abort(404)

    if folder not in ALLOWED_VIZ_FOLDERS:
        sec_log.warning(f"INVALID_FOLDER ip={request.remote_addr} folder={folder!r}")
        abort(404)

    safe_name = os.path.basename(filename)          # strip any path components
    path = safe_result_path(RESULT_FOLDER, job_id, folder, safe_name)
    if path is None or not os.path.isfile(path):
        abort(404)

    return send_from_directory(os.path.dirname(path), safe_name)


@app.route('/download/<job_id>/<filename>')
def download_file(job_id, filename):
    """Serve a downloadable result file. Validates job_id and filename."""

    if not is_valid_uuid(job_id):
        abort(404)

    safe_name = os.path.basename(filename)

    if safe_name not in ALLOWED_DOWNLOAD_FILES:
        sec_log.warning(
            f"INVALID_DOWNLOAD ip={request.remote_addr} "
            f"job={job_id} file={filename!r}"
        )
        abort(404)

    path = safe_result_path(RESULT_FOLDER, job_id, safe_name)
    if path is None or not os.path.isfile(path):
        abort(404)

    return send_file(path, as_attachment=True)


# ---------------------------------------------------------------------------
# Filter endpoint (stateless — reads no user-supplied file paths)
# ---------------------------------------------------------------------------

@app.route('/filter')
def filter_topics():
    """
    Topic keyword filter. Note: this endpoint is stateless and does not
    expose a file path — it reads no job-specific file.
    TODO: wire to a job-scoped topics.txt once session handling is added.
    """
    query   = request.args.get('query', '').lower()
    results = []
    # Placeholder: returns empty until job-aware session is implemented.
    return jsonify(results)


# ---------------------------------------------------------------------------
# Static pages
# ---------------------------------------------------------------------------

@app.route('/about')
def about():   return render_template('about.html')

@app.route('/docs')
def docs():    return render_template('docs.html')

@app.route('/contact')
def contact(): return render_template('contact.html')

@app.route('/sitemap')
def sitemap(): return render_template('sitemap.html')

@app.route('/privacy')
def privacy(): return render_template('privacy.html')

@app.route('/terms')
def terms():   return render_template('terms.html')


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

@app.errorhandler(404)
def handle_404(e):
    return render_template("error.html", error={
        'code': 404,
        'name': 'Page Not Found',
        'description': 'The page you requested could not be found.',
    }), 404

@app.errorhandler(429)
def handle_429(e):
    sec_log.warning(f"RATE_LIMITED ip={request.remote_addr}")
    return render_template("error.html", error={
        'code': 429,
        'name': 'Too Many Requests',
        'description': 'You have submitted too many requests. Please wait a moment and try again.',
    }), 429

@app.errorhandler(500)
def handle_500(e):
    return render_template("error.html", error={
        'code': 500,
        'name': 'Internal Server Error',
        'description': 'Something went wrong on our end. Please try again later.',
    }), 500

@app.errorhandler(503)
def handle_503(e):
    return render_template("error.html", error={
        'code': 503,
        'name': 'Server Busy',
        'description': 'The server is temporarily unavailable. Please try again shortly.',
    }), 503

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)