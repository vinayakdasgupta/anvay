# -*- coding: utf-8 -*-

import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler("app.log", encoding="utf-8")]
)

from flask import Flask, request, render_template, send_from_directory
import os
import re
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from werkzeug.utils import secure_filename
from collections import Counter
import string
import csv
from viz import *
from csvSave import *
import pyLDAvis
import pyLDAvis.gensim_models

# Flask application
app = Flask(__name__)
app.secret_key = 'abc123'  # Needed for using sessions
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB

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

def allowed_file(filename):
    """
    Check if the file has an allowed extension.
    Returns True if the file has an allowed extension, else False.
    """
    # List of allowed file extensions
    ALLOWED_EXTENSIONS = {'txt'}
    
    # Check if the file has an extension and if it is in the allowed list
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['STOPWORDS_FOLDER'] = STOPWORDS_FOLDER

# Load noun roots, verb roots, noun suffixes, and verb suffixes
NOUN_ROOTS_FILE = os.path.join(app.config['STOPWORDS_FOLDER'], 'noun_roots.txt')
VERB_ROOTS_FILE = os.path.join(app.config['STOPWORDS_FOLDER'], 'verb_roots.txt')
NOUN_SUFFIXES_FILE = os.path.join(app.config['STOPWORDS_FOLDER'], 'noun_suffixes.txt')
VERB_SUFFIXES_FILE = os.path.join(app.config['STOPWORDS_FOLDER'], 'verb_suffixes.txt')

NOUN_ROOTS = set(load_list_from_file(NOUN_ROOTS_FILE))
VERB_ROOTS = set(load_list_from_file(VERB_ROOTS_FILE))
NOUN_SUFFIXES = load_list_from_file(NOUN_SUFFIXES_FILE)
VERB_SUFFIXES = load_list_from_file(VERB_SUFFIXES_FILE)

# Create necessary directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(STOPWORDS_FOLDER, exist_ok=True)


# Load Bengali stopwords from NLTK
stop_words = set(stopwords.words('bengali'))


# Helper functions for the rule-based stemmer
def is_valid_root(word, root_dict):
    return word in root_dict


def get_word_pos(word):
    """
    Determine the part of speech (POS) of a word.
    Returns "NOUN" or "VERB". Modify with a more robust POS tagger if needed.
    """
    if word.endswith(tuple(NOUN_SUFFIXES)):
        return "NOUN"
    elif word.endswith(tuple(VERB_SUFFIXES)):
        return "VERB"
    else:
        return "NOUN"  # Default to NOUN

def remove_suffix(word, suffix):
    if word.endswith(suffix):
        return word[:-len(suffix)]
    return word

def bengali_stemmer(word):
    """
    Perform rule-based stemming for Bengali words.
    """
    WPOS = get_word_pos(word)

    # Process nouns
    if WPOS == "NOUN":
        if is_valid_root(word, NOUN_ROOTS):
            return word  # Return as-is if it's a valid noun root
        for suffix in NOUN_SUFFIXES:
            possible_root = remove_suffix(word, suffix)
            if is_valid_root(possible_root, NOUN_ROOTS):
                return possible_root

    # Process verbs
    elif WPOS == "VERB":
        if is_valid_root(word, VERB_ROOTS):
            return word  # Return as-is if it's a valid verb root
        for suffix in VERB_SUFFIXES:
            possible_root = remove_suffix(word, suffix)
            if is_valid_root(possible_root, VERB_ROOTS):
                return possible_root

    # If no valid root is found, return the original word
    return word

# Function to remove punctuation, including Bengali-specific punctuation
def remove_punctuation(text):
    bengali_punctuations = "।॥“”‘’"
    all_punctuations = string.punctuation + bengali_punctuations
    text = re.sub(f"[{re.escape(all_punctuations)}]", "", text)
    return text

# Function to apply stemming to tokens using the custom stemmer
def stem_tokens(tokens_list):
    return [[bengali_stemmer(token) for token in tokens] for tokens in tokens_list]

# Function to load custom stopwords from a text file
def load_stopwords(file_path):
    if not os.path.exists(file_path):
        return set()
    with open(file_path, 'r', encoding='utf-8') as f:
        return set(line.strip() for line in f.readlines())

# Keep only Bengali words    
def filter_non_bengali_words(tokens):
    """
    Retain only Bengali words in the token list.
    """
    bengali_pattern = re.compile(r'[\u0980-\u09FF]+')  # Matches Bengali characters
    return [token for token in tokens if bengali_pattern.match(token)]

# Function to remove most frequent words based on percentage
def remove_most_frequent_words(all_tokens, percent_most_common):
    flat_tokens = [token for tokens in all_tokens for token in tokens]
    word_freq = Counter(flat_tokens)
    total_words = len(flat_tokens)
    if total_words == 0:
        return all_tokens
    num_to_remove = int((percent_most_common / 100) * len(word_freq))
    if percent_most_common == 100:
        return [[] for _ in all_tokens]
    most_common_words = set([word for word, _ in word_freq.most_common(num_to_remove)])
    filtered_tokens = []
    for tokens in all_tokens:
        filtered_tokens.append([token for token in tokens if token not in most_common_words])
    return filtered_tokens

# Process and prepare LDA models (same as original)
def process_txt_files(file_paths, num_topics, iterations, chunk_size, ngram, alpha, per_word_topics, no_below, no_above, use_stemming, use_multicore, percent_most_common):
    all_tokens = []
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            text = remove_punctuation(text)
        tokens = word_tokenize(text)
        tokens = filter_non_bengali_words(tokens)  
        tokens = [token for token in tokens if len(token) > 1]
        tokens = [word for word in tokens if word not in stop_words]
        if use_stemming:
            tokens = stem_tokens([tokens])[0]
        all_tokens.append(tokens)
    if percent_most_common > 0:
        all_tokens = remove_most_frequent_words(all_tokens, percent_most_common)
        return all_tokens
 
def apply_ngrams(all_tokens, ngram):
    """
    Applies n-gram modeling (bigram or trigram) to tokenized texts.
    """
    if ngram == 'bigram':
        bigram_mod = gensim.models.Phrases(all_tokens, min_count=2, threshold=50)
        tokens_ngrams = [bigram_mod[tokens] for tokens in all_tokens]
    elif ngram == 'trigram':
        bigram_mod = gensim.models.Phrases(all_tokens, min_count=2, threshold=50)
        trigram_mod = gensim.models.Phrases(bigram_mod[all_tokens], min_count=2, threshold=50)
        tokens_ngrams = [trigram_mod[bigram_mod[tokens]] for tokens in all_tokens]
    else:
        tokens_ngrams = all_tokens
    return tokens_ngrams

def build_lda_model(all_tokens, num_topics, chunk_size, iterations, update_every, alpha, per_word_topics, no_below, no_above, use_multicore):
    """
    Builds an LDA model from the tokenized texts.
    """
    id2word = corpora.Dictionary(all_tokens)
    id2word.filter_extremes(no_below=no_below, no_above=no_above)
    corpus = [id2word.doc2bow(tokens) for tokens in all_tokens]

    logging.debug("Checking for empty documents in the corpus...")
    for i, bow in enumerate(corpus):
        if len(bow) == 0:
            logging.debug(f"Document {i} has no tokens in the corpus. Tokens: {all_tokens[i]}")
    logging.debug("Finished checking for empty documents.")
   

    if use_multicore:
        # Use LdaMulticore for parallel processing (no update_every parameter)
        lda_model = gensim.models.LdaMulticore(
            corpus=corpus,
            id2word=id2word,
            num_topics=num_topics,
            random_state=100,
            chunksize=chunk_size,
            passes=iterations,
            alpha=alpha,
            per_word_topics=per_word_topics,
            workers=os.cpu_count() - 1
        )
    else:
        # Use LdaModel with update_every parameter
        lda_model = gensim.models.LdaModel(
            corpus=corpus,
            id2word=id2word,
            num_topics=num_topics,
            random_state=100,
            chunksize=chunk_size,
            passes=iterations,
            update_every=update_every,  # Use update_every here
            alpha=alpha,
            per_word_topics=per_word_topics,
        )

    # Debugging: Check topic distributions for each document
    for i, bow in enumerate(corpus):
        topic_distribution = lda_model.get_document_topics(bow, minimum_probability=0.0)
        total_weight = sum(weight for _, weight in topic_distribution)
        logging.debug(f"Document {i} topic distribution: {topic_distribution}, Total weight: {total_weight}")
        if total_weight < 0.01:  # Example threshold for negligible distribution
            logging.debug(f"Document {i} has negligible topic distribution. Tokens: {all_tokens[i]}")

    return lda_model, corpus, id2word, all_tokens

@app.route('/', methods=['GET', 'POST'])
def upload_files():
    if request.method == 'POST':
        files = request.files.getlist("files")
        if not files:
            return "No files uploaded.", 400
        for file in files:
            if file and allowed_file(file.filename):
                file_path = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
                file.save(file_path)
        return "Files uploaded successfully.", 200
    return render_template('upload.html')

@app.route('/process', methods=['POST'])
def process_files():
    if 'files[]' not in request.files:
        return "No files uploaded", 400

    files = request.files.getlist('files[]')
    if not files or files[0].filename == '':
        return "No files selected", 400

    # Retrieve LDA model parameters
    num_topics = int(request.form.get('num_topics', 10))
    iterations = int(request.form.get('iterations', 40))
    chunk_size = int(request.form.get('chunk_size', 200))
    update_every = int(request.form.get('update_every', 1))
    ngram = request.form.get('ngram', 'unigram')
    alpha = request.form.get('alpha', 'asymmetric')
    per_word_topics = request.form.get('per_word_topics', 'false').lower() == 'true'
    no_below = int(request.form.get('no_below', 1))
    no_above = float(request.form.get('no_above', 0.9))

    use_stemming = request.form.get('use_stemming', 'false').lower() == 'true'
    use_multicore = request.form.get('use_multicore', 'false').lower() == 'true'
    percent_most_common = float(request.form.get('percent_most_common', 0))

    # Handle custom stopwords file upload
    custom_stopwords_file = request.files.get('custom_stopwords')
    if custom_stopwords_file and custom_stopwords_file.filename.endswith('.txt'):
        custom_stopwords_path = os.path.join(STOPWORDS_FOLDER, custom_stopwords_file.filename)
        custom_stopwords_file.save(custom_stopwords_path)
        custom_stopwords = load_stopwords(custom_stopwords_path)
        stop_words.update(custom_stopwords)

    # Save uploaded files
    file_paths = []
    for file in files:
        if not file.filename.endswith('.txt'):
            return "Invalid file format. Please upload .txt files only.", 400

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        file_paths.append(file_path)

    # Process text files to generate tokens
    all_tokens = []
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            text = remove_punctuation(text)
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if len(token) > 1]
        tokens = [word for word in tokens if word not in stop_words]
        if use_stemming:
            tokens = stem_tokens([tokens])[0]
        all_tokens.append(tokens)

    print(f"Tokens after preprocessing for document: {tokens}")

    # Remove most frequent words
    if percent_most_common > 0:
        all_tokens = remove_most_frequent_words(all_tokens, percent_most_common)

    # Apply n-grams
    all_tokens = apply_ngrams(all_tokens, ngram)

    # Build LDA model
    lda_model, corpus, id2word, _ = build_lda_model(
        all_tokens,
        num_topics,
        chunk_size,
        iterations,
        update_every,
        alpha,
        per_word_topics,
        no_below,
        no_above,
        use_multicore
    )

    coherence_model = CoherenceModel(model=lda_model, corpus=corpus, dictionary=id2word, texts=all_tokens, coherence='c_v')
    coherence_score = coherence_model.get_coherence()

# Generate visualizations
    visualizations = {}
    try:
        pyldavis_path = os.path.join(RESULT_FOLDER, "pyldavis.html")
        pyldavis_output = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)
        pyLDAvis.save_html(pyldavis_output, pyldavis_path)
        visualizations["pyldavis"] = ("PyLDAvis", "pyldavis.html")

        bokeh_paths = create_bokeh_visualizations(lda_model, corpus, id2word)
        visualizations["scatter"] = ("Scatter Plot", bokeh_paths.get("scatter"))
        visualizations["bars"] = ("Bar Chart", bokeh_paths.get("bars"))

        heatmap_path = create_heatmap(lda_model)
        visualizations["heatmap"] = ("Heatmap", heatmap_path)

        evolution_path = create_topic_evolution(lda_model, corpus)
        visualizations["evolution"] = ("Topic Evolution", evolution_path)

        chord_path = create_chord_diagram(lda_model)
        visualizations["chord"] = ("Chord Diagram", chord_path)

        clustering_path = create_hierarchical_clustering(lda_model)
        visualizations["clustering"] = ("Hierarchical Clustering", clustering_path)

        distribution_path = create_topic_distribution_per_document(lda_model, corpus)
        visualizations["distribution"] = ("Topic Distribution", distribution_path)

        coherence_chart_path = create_coherence_bar_chart(lda_model, all_tokens, id2word, output_dir=RESULT_FOLDER)
        visualizations["coherence_chart"] = ("Topic Coherence Bar Chart", coherence_chart_path)



                # Save topics to text and CSV
        txt_path, csv_path = save_topics_to_files(lda_model, num_topics)
        visualizations["topics_txt"] = ("Topics Text File", os.path.basename(txt_path))
        visualizations["topics_csv"] = ("Topics CSV File", os.path.basename(csv_path))



        
    except Exception as e:
        logging.error(f"Error generating visualizations: {e}")
        return "Error generating visualizations.", 500

    return render_template('result.html', visualizations=visualizations)

@app.route('/results/<path:filename>')
def download_file(filename):
    return send_from_directory(RESULT_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)