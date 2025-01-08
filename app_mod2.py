# -*- coding: utf-8 -*-
from flask import Flask, request, render_template, send_file, send_from_directory, jsonify
import os
import re
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.data import find
import csv
import string
from collections import Counter

def download_nltk_resources():
    try:
        find('corpora/stopwords')
        find('tokenizers/punkt')
    except LookupError:
        nltk.download('stopwords')
        nltk.download('punkt')

# Call this function before starting the app
download_nltk_resources()

# Flask application
app = Flask(__name__)

# Function to load lists from text files
def load_list_from_file(file_path):
    if not os.path.exists(file_path):
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines() if line.strip()]

# Folder locations for file handling
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
STOPWORDS_FOLDER = 'stopwords'
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
        tokens = [token for token in tokens if len(token) > 1]
        tokens = [word for word in tokens if word not in stop_words]
        if use_stemming:
            tokens = stem_tokens([tokens])[0]
        all_tokens.append(tokens)
    if percent_most_common > 0:
        all_tokens = remove_most_frequent_words(all_tokens, percent_most_common)
    if ngram == 'bigram':
        bigram_mod = gensim.models.Phrases(all_tokens, min_count=2, threshold=50)
        tokens_ngrams = [bigram_mod[tokens] for tokens in all_tokens]
    elif ngram == 'trigram':
        bigram_mod = gensim.models.Phrases(all_tokens, min_count=2, threshold=50)
        trigram_mod = gensim.models.Phrases(bigram_mod[all_tokens], min_count=2, threshold=50)
        tokens_ngrams = [trigram_mod[bigram_mod[tokens]] for tokens in all_tokens]
    else:
        tokens_ngrams = all_tokens
    id2word = corpora.Dictionary(tokens_ngrams)
    id2word.filter_extremes(no_below=no_below, no_above=no_above)
    corpus = [id2word.doc2bow(tokens) for tokens in tokens_ngrams]
    lda_model = gensim.models.LdaMulticore(corpus=corpus, id2word=id2word, num_topics=num_topics, random_state=100,
                                           chunksize=chunk_size, passes=iterations, alpha=alpha, per_word_topics=per_word_topics,
                                           workers=os.cpu_count() - 1 if use_multicore else None)
    vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
    vis_path = os.path.join(RESULT_FOLDER, 'lda_visualization.html')
    pyLDAvis.save_html(vis, vis_path)
    topics = lda_model.show_topics(num_topics=num_topics, formatted=False)
    topics_data = [{"Topic": i, "Words": [word[0] for word in topic]} for i, topic in topics]
    txt_path = os.path.join(RESULT_FOLDER, 'topics.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        for topic in topics_data:
            f.write(f"Topic {topic['Topic']}:\n")
            f.write(", ".join(topic['Words']) + "\n\n")
    csv_path = os.path.join(RESULT_FOLDER, 'topics.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Topic', 'Words'])
        for topic in topics_data:
            writer.writerow([topic['Topic'], ", ".join(topic['Words'])])
    return vis_path, txt_path, csv_path, lda_model, corpus, id2word, all_tokens


@app.route('/')
def upload_file():
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
    ngram = request.form.get('ngram', 'unigram')
    alpha = request.form.get('alpha', 'asymmetric')
    per_word_topics = request.form.get('per_word_topics', 'false').lower() == 'true'
    no_below = int(request.form.get('no_below', 1))
    no_above = float(request.form.get('no_above', 0.9))

    # Check if stemming is enabled
    use_stemming = request.form.get('use_stemming', 'false').lower() == 'true'

    # Check if multicore processing is enabled
    use_multicore = request.form.get('use_multicore', 'false').lower() == 'true'

    # Get percentage of most common words to remove
    percent_most_common = float(request.form.get('percent_most_common', 0))

    # Handle custom stopword file upload
    custom_stopwords_file = request.files.get('custom_stopwords')
    if custom_stopwords_file and custom_stopwords_file.filename.endswith('.txt'):
        custom_stopwords_path = os.path.join(STOPWORDS_FOLDER, custom_stopwords_file.filename)
        custom_stopwords_file.save(custom_stopwords_path)
        custom_stopwords = load_stopwords(custom_stopwords_path)
        stop_words.update(custom_stopwords)

    # Check if stopwords removal is enabled
    remove_stopwords = 'remove_stopwords' in request.form

    # Save uploaded files and process them
    file_paths = []
    for file in files:
        if not file.filename.endswith('.txt'):
            return "Invalid file format. Please upload .txt files only.", 400

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        file_paths.append(file_path)

    # Process files and generate LDA visualization
    vis_path, txt_path, csv_path, lda_model, corpus, id2word, all_tokens = process_txt_files(
        file_paths, num_topics, iterations, chunk_size, ngram, alpha, per_word_topics, no_below, no_above, use_stemming, use_multicore, percent_most_common
    )

    # Remove stopwords if the checkbox was checked
    if remove_stopwords:
        all_tokens = [[token for token in tokens if token not in stop_words] for tokens in all_tokens]

    # Calculate coherence score
    cm = CoherenceModel(model=lda_model, corpus=corpus, dictionary=id2word, texts=all_tokens, coherence='c_v')
    coherence_score = cm.get_coherence()

    # Pass the results and coherence score to the result template
    return render_template('result.html', vis_path=vis_path, txt_path=txt_path, csv_path=csv_path, coherence_score=coherence_score)


@app.route('/results/lda_visualization.html')
def serve_visualization():
    return send_from_directory('results', 'lda_visualization.html')

@app.route('/download/<file_type>', methods=['GET'])
def download_topics(file_type):
    file_map = {
        'txt': 'topics.txt',
        'csv': 'topics.csv'
    }
    if file_type not in file_map:
        return "Invalid file type", 400

    file_path = os.path.join(RESULT_FOLDER, file_map[file_type])
    if not os.path.exists(file_path):
        return "File not found", 404

    return send_file(file_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
