# Anvay: Bengali LDA Topic Modeling Application

## Overview

Anvay is a Flask-based Bengali text processing and topic modeling tool that uses Latent Dirichlet Allocation (LDA) to extract topics from uploaded text files. The application includes a custom rule-based Bengali stemmer, stopword removal, and support for n-grams and additional preprocessing features. It generates a visualization of topics and allows exporting the results in .txt and .csv formats.

## Features

### File Upload:

- Accepts multiple .txt files for processing.

### Text Preprocessing:

- Removes stopwords (Bengali stopwords from NLTK and custom stopwords).
- Removes punctuation (including Bengali-specific punctuation).
- Rule-based stemming for Bengali nouns and verbs.
- Removes most frequent words based on user-defined thresholds.

### Topic Modeling:

- Customizable LDA model parameters (number of topics, iterations, etc.).
- Support for unigrams, bigrams, and trigrams.
- Generates an interactive HTML visualization of topics using pyLDAvis.

### Result Export:

- Topics can be downloaded as .txt or .csv files.

### Coherence Score:

- Evaluates the coherence of the generated topics using the gensim library.

### Visualizations:

Anvay supports the following visualizations:

- **PyLDAvis**: Interactive visualization of topics and their relationships.
- **Scatter Plot**: Visual representation of topic distribution using Bokeh.
- **Bar Chart**: Displays the frequency of topics.
- **Heatmap**: Illustrates the relationships and strengths between topics.
- **Chord Diagram**: Visualizes connections between topics.
- **Hierarchical Clustering**: Groups topics based on similarity.
- **Topic Evolution Chart**: Shows how topics evolve over documents.
- **Topic Distribution Per Document**: Provides insight into topic coverage within individual documents.
- **Coherence Bar Chart**: Evaluates the coherence of topics for better understanding.

## Requirements

- Python 3.8+
- Flask
- Gensim
- NLTK
- PyLDAvis
- Bokeh **2.4.3** (Required)
- Matplotlib
- Other dependencies as specified in `requirements.txt`

**Note:** The application requires Bokeh version 2.4.3. Using any other version may lead to compatibility issues.

### Python Libraries

The following Python libraries are required:

- Flask
- Gensim
- PyLDAvis
- NLTK
- re
- csv
- string
- os
- collections

Install the dependencies using:

```bash
pip install Flask gensim PyLDAvis nltk
```

## Getting Started

### 1. Setup

Clone the repository and navigate to the project directory. Check that the necessary folders for file uploads, results, and stopwords are there:

```bash
git clone https://github.com/vinayakdasgupta/anvay/
```

### 2. NLTK Resources

The application automatically downloads required NLTK resources (e.g., stopwords and punkt). If needed, you can manually download them:

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

### 3. Running the Application

Start the Flask application:

```bash
python app.py
```

The application will run at `http://127.0.0.1:5000/`.

## Using the Application

### 1. Upload Files

Navigate to the homepage and upload .txt files for topic modeling. Optionally, upload a custom stopwords file (a .txt file with one word per line).

### 2. Configure Parameters

Set the following parameters:

#### LDA Model Parameters:

- `num_topics`: Number of topics to extract.
- `iterations`: Number of iterations for model training.
- `chunk_size`: Size of chunks to process at a time.
- `alpha`: Dirichlet prior for topic distribution.

#### Preprocessing Options:

- Enable/disable stemming.
- Set the percentage of most frequent words to remove.
- Choose n-gram type: unigram, bigram, or trigram.
- Enable/disable stopword removal and multicore processing.

### 3. View and Download Results

- View the interactive topic visualization.
- Download the topics as .txt or .csv files.

## Folder Structure

- `uploads/`: Stores uploaded text files.
- `results/`: Stores the LDA visualization (`lda_visualization.html`) and generated topic files (`topics.txt`, `topics.csv`).
- `stopwords/`: Stores custom stopwords files.

## Customization

### Adding Custom Stopwords

Add your custom stopwords file in the `stopwords/` folder or upload it through the application. The file should contain one word per line.

### Modifying the Rule-Based Stemmer

Edit the noun and verb root/suffix files in the `stopwords/` folder:

- `noun_roots.txt`
- `verb_roots.txt`
- `noun_suffixes.txt`
- `verb_suffixes.txt`

## Example Workflow

1. Upload Bengali text files (.txt).
2. Configure preprocessing (e.g., stemming, stopword removal, etc.).
3. Set LDA model parameters (e.g., number of topics, iterations).
4. Click Process to generate the topics.
5. View the interactive topic visualization or download the results.

## Future Enhancements

- Integrate a more robust POS tagger for Bengali.
- Improve the stemming process with a hybrid approach (rule-based + statistical methods).
- Improve memory handling.
- Create interactive JS visualizations for static images.

## Known Issues

- **File Upload Limitation:** The total file size and the number of files that can be uploaded depend on Flask’s internal limitations. The application has been tested with up to 980 files successfully.
- **Threading Issue:** Flask may sometimes encounter threading issues under heavy load.

## License

This application is open-source and available under the MIT License. See `LICENSE` for details.

## Acknowledgments

- Gensim: Topic modeling and coherence evaluation.

