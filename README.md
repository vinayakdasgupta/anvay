# anvay: A Topic Modelling Dashboard



https://github.com/user-attachments/assets/75327a2f-27fb-467a-8ebf-e1585a97e0ec



**anvay** is a web-based topic modelling interface built for exploring, analysing, and interpreting large corpora of text. Developed with a focus on literary and historical materials, anvay offers users fine-grained control over preprocessing options and presents results in a structured, interactive interface designed for both researchers and students. The application is modular, interpretable, and lightweight, making it suitable for public deployment and pedagogical use.

## Overview

anvay takes plain-text `.txt` files and performs preprocessing (tokenisation, stemming, stopword removal, frequency filtering, n-gram construction), builds a Latent Dirichlet Allocation (LDA) topic model using Gensim, and visualises the results across multiple tabs with topic-wise document insights.

The interface is designed to foreground interpretability over complexity: there is no reliance on neural networks, transformer embeddings, or LLMs. Every transformation is documented and controlled by the user.

---

Release 2.0 (in progress)
-------------------------

This release represents a significant architectural and methodological expansion of anvay. While Bengali remains the primary and most stable language, anvay now supports multiple languages through a unified, language-aware preprocessing pipeline. Non-Bengali languages should be considered **experimental** at this stage.

### Multilingual Expansion (Experimental)

anvay now supports topic modelling for multiple languages beyond Bengali:

*   **English**: Tokenisation with NLTK, lemmatisation using WordNet. Normalisation order is fixed (filter → lemmatise) to ensure predictable behaviour.
    
*   **Hindi**: Custom Devanagari tokenisation with dictionary-based lemmatisation (via simplemma). Stopword handling and normalisation are currently conservative and experimental.
    
*   **Tamil**: Custom Unicode-aware tokenisation with dictionary-based stemming.Designed for narrative and devotional corpora; behaviour is still under evaluation.
    

All non-Bengali language pipelines are modular and isolated, allowing future refinement without affecting core Bengali functionality.

### Expanded Bengali Stemming Dictionary

*   The Bengali stemming dictionary has been expanded from approximately **7,000** word pairs to over **75,000** word pairs.
    
*   This significantly improves coverage of inflectional and derivational variants.
    
*   Results in more stable and interpretable topic-word distributions, especially for long-form journalistic and literary corpora.
    

### Configurable Stemming and Filtering Order

*   Users can now explicitly choose the preprocessing order:
    
    *   **Stem → Filter**
        
    *   **Filter → Stem**
        
*   This makes preprocessing behaviour transparent and reproducible.
    
*   Enables comparative experimentation with token reduction strategies and their effects on topic coherence and interpretability.
    
*   Particularly relevant for dictionary-based stemming pipelines.
    

### Enhanced Representative Sentence Display

*   For each topic, the interface now displays:
    
    *   The most representative sentence.
        
    *   Two sentences before and two sentences after it (context window).
        
    *   The topic weight of the representative sentence.
        
*   This improves interpretability by situating topic salience within **local textual context**, rather than presenting isolated sentences.
    

### Notes

*   Bengali remains the most mature and thoroughly validated language in anvay.
    
*   English, Hindi, and Tamil support are **experimental** and intended primarily for methodological exploration and testing.
    
*   This release prioritises architectural extensibility and interpretive transparency over feature completeness.

### Release 1.1.1
#### Clustering
- Enhanced hierarchical clustering with BERTopic-style merged-cluster keyword tooltips.  

#### Documentation
- Updated documentation to explain that Heatmap and Bar Chart visualise the same topic–word weight matrix.  

#### Visualisation  
- Added top-word hover tooltips across all visualisations for clearer topic interpretation.  
- Standardised global topic colour scheme across all charts. The automated marker colours are determined by golden-ratio hue jumping + lightness alternation  
- Reduced number of displayed terms in plots to prevent hidden tick labels; added hover-based x-axis details where needed.  
- Unified Plotly font styling using Roboto/Noto Bengali; reduced margins for a cleaner layout.  

#### Quality-Of-Life
- Clarified Topic Evolution axis (document upload order) and added filenames to hover output.  
- Added missing loading spinner to indicate processing during analysis.  

---
### Notes
- Hovering on Plotly legends is unfortunately not supported; tooltips are therefore provided directly on the plots.

---
These changes significantly improve clarity, consistency, and user experience in the visualisation interface.

## Features

### Upload & Preprocessing
- Upload up to **800 UTF-8 encoded .txt files** at once (maximum total size: 100MB)
- Corpus size and token thresholds enforced to ensure browser responsiveness
- Preprocessing controls include:
  - Standard + custom Bengali stopwords
  - `no_below` and `no_above` frequency thresholds
  - Top-N% most frequent tokens filter
  - N-gram selection: unigrams, bigrams, trigrams
  - Dictionary-based stemming

### Dictionary-Based Stemming (v1.1.0)
- Replaces earlier rule-based suffix stripping
- Offers better semantic interpretability and topic-word clarity

### Topic Modelling
- Gensim's `LdaMulticore` implementation for fast, multicore topic modelling
- Tunable parameters:
  - Number of topics
  - Passes and iterations
  - Alpha and Eta priors
  - Chunk size
  - Minimum probability threshold


### Visualisations (Tabbed UI)
- **Visualisations Tab**: Bar chart, scatter plot, pie chart, heatmap, topic-word network graph
- **Report Tab**: Training summary, top tokens, topic prevalence, representative documents
- **Downloads Tab**: Export results as CSV and TXT
- **Guide Tab**: Step-by-step interpretive instructions

### Topic-Document Drilldown
- Per-topic list of most representative documents
- Context-aware sentence preview
- Topic label and confidence indicator

### Design Principles
- Mobile-friendly and responsive layout

---

## Documentation



anvay includes a fully integrated documentation panel accessible from the interface itself. The documentation is designed not merely as technical reference, but as a pedagogical aid that walks users through each stage of the topic modelling process — from corpus preparation and parameter selection to result interpretation. It explains preprocessing choices (e.g. stopword filtering, n-gram selection, stemming) in clear language, and provides visual examples and tooltips to guide first-time users. The documentation also includes a walkthrough of a sample run, highlighting what users can expect from the model outputs. Importantly, the documentation assumes no prior knowledge of machine learning, making anvay accessible to students, scholars, and corpus curators working with Bengali texts.

---

## Screenshots

<details open>
  <summary>Upload interface screenshot</summary>
  <img src="docs/img/anvay_upload.PNG" alt="Upload interface" width="500"/>
</details>

<details open>
  <summary>Documentation interface screenshot</summary>
  <img src="docs/img/anvay_documentation.PNG" alt="Upload interface" width="500"/>
</details>

<details open>
  <summary>Visualization interface screenshot</summary>
  <img src="docs/img/anvay_viz.PNG" alt="Upload interface" width="500"/>
</details>

<details open>
  <summary>Report interface screenshot</summary>
  <img src="docs/img/anvay_results.PNG" alt="Upload interface" width="500"/>
</details>


---

## Technical Stack

- **Backend**: Python (Flask), Gensim, NLTK, NetworkX, Scikit-learn
- **Frontend**: Bootstrap, jQuery, Plotly, Bokeh, Seaborn
- **Deployment**: Designed to be hosted on a university or personal server (e.g., via Gunicorn)

---

## Installation

**anvay** has been tested with Python 3.9-3.11 and Gensim 4.3.x.  
Two installation methods are provided:

- **Standard installation (virtual environment)** 
- **Docker-based installation** 

---

## Option 1: Standard installation (virtual environment)

This method installs anvay directly on your system using a Python virtual environment.

### Prerequisites
- Python 3.9-3.11
- Git  
- pip (Python package installer)

### Step-by-step instructions

Clone the repository:

```bash
git clone https://github.com/vinayakdasgupta/anvay.git
cd anvay
```

Create and activate a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python app.py
```

Access the web interface:

```
http://localhost:5000
```

You can now upload `.txt` files and begin exploring topics.

---

## Option 2: Docker-based installation (recommended for reproducibility)


### Prerequisites
- Docker (Docker Desktop on Windows/macOS)

### Step-by-step instructions

Clone the repository:

```bash
git clone https://github.com/vinayakdasgupta/anvay.git
cd anvay
```

Build the Docker image:

```bash
docker build -t anvay .
```

Run the container:

```bash
docker run -p 5000:5000 anvay
```

Access the web interface:

```
http://localhost:5000
```

You can now upload .txt files and begin exploring topics.

---

## How to Cite anvay

If you use anvay in academic work, please cite it as follows:

> Das Gupta, Vinayak. *anvay: a web-based tool for interpretive topic modelling in bengali*.  
Zenodo. https://doi.org/10.5281/zenodo.18186215

Once a DOI or formal publication is available, this should be replaced with the appropriate citation.

---

## Referenced Datasets and Libraries

The following tools, datasets, and libraries are used in anvay and should be cited as appropriate:

### Lemmatization Dataset
```bibtex
@inproceedings{chakrabarty-etal-2017-context,
    title = "Context Sensitive Lemmatization Using Two Successive Bidirectional Gated Recurrent Networks",
    author = "Chakrabarty, Abhisek  and
      Pandit, Onkar Arun  and
      Garain, Utpal",
    editor = "Barzilay, Regina  and
      Kan, Min-Yen",
    booktitle = "Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2017",
    address = "Vancouver, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P17-1136/",
    doi = "10.18653/v1/P17-1136",
    pages = "1481--1491"
}

@article{alam2021review,
  title={A Review of Bangla Natural Language Processing Tasks and the Utility of Transformer Models},
  author={Alam, Firoj and Hasan, Md Arid and Alam, Tanvir and Khan, Akib and Tajrin, Janntatul and Khan, Naira and Chowdhury, Shammur Absar},
  journal={arXiv preprint arXiv:2107.03844},
  year={2021}
}

@article{islam2025banglalem,
  title={BanglaLem: A Transformer-based Bangla Lemmatizer with an Enhanced Dataset},
  author={Islam, Md Fuadul and Hasan, Jakir and Islam, Md Ashikul and Dewan, Prato and Rahman, M Shahidur},
  journal={Systems and Soft Computing},
  pages={200244},
  year={2025},
  publisher={Elsevier}
}
```

### Gensim
```bibtex
@inproceedings{rehurek_lrec,
  author = {Řehůřek, Radim and Sojka, Petr},
  title = {Software Framework for Topic Modelling with Large Corpora},
  booktitle = {Proceedings of the LREC 2010 Workshop on New Challenges for NLP Frameworks},
  pages = {45--50},
  year = {2010},
  publisher = {ELRA},
  address = {Valletta, Malta},
}
```

### NLTK
```bibtex
@book{bird2009natural,
  author = {Bird, Steven and Klein, Ewan and Loper, Edward},
  title = {Natural Language Processing with Python},
  year = {2009},
  publisher = {O'Reilly Media, Inc.}
}
```

### NetworkX
```bibtex
@inproceedings{hagberg2008exploring,
  author = {Hagberg, Aric A. and Schult, Daniel A. and Swart, Pieter J.},
  title = {Exploring Network Structure, Dynamics, and Function using NetworkX},
  booktitle = {Proceedings of the 7th Python in Science Conference (SciPy2008)},
  pages = {11--15},
  year = {2008}
}
```

### Scikit-learn
```bibtex
@article{pedregosa2011scikit,
  author = {Pedregosa, Fabian et al.},
  title = {Scikit-learn: Machine Learning in Python},
  journal = {Journal of Machine Learning Research},
  volume = {12},
  pages = {2825--2830},
  year = {2011}
}
```

### Plotly
> Plotly Technologies Inc. (2015). *Collaborative data science*. Montreal, QC. https://plot.ly

---

## Acknowledgements

anvay draws on multiple open-source projects:
- **Gensim** – topic modelling
- **NLTK** – stopword filtering
- **Plotly, Seaborn, Matplotlib** – visualisation
- **NetworkX** – topic-word graph
- **Scikit-learn** – PCA and clustering
- **Flask** – web application framework

---

## License

anvay is released under the MIT License. 

---

## Contact

Vinayak Das Gupta  
Shiv Nadar University
[https://vinayakdasgupta.com]

For questions, suggestions, or scholarly collaborations, please open an issue or contact via GitHub.

