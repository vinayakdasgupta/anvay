# anvay

**anvay** is a web-based application for topic modelling Bengali text corpora using Latent Dirichlet Allocation (LDA). It is designed for interpretive use by researchers, students, and digital humanists, and offers a streamlined interface for corpus upload, parameter tuning, and interactive exploration of results.

This release provides a production-ready implementation of the full pipeline — from corpus ingestion to multi-format visualisation — with custom preprocessing support for Bengali.

## Features

- Upload plain text files (.txt) as corpus input
- Configure LDA parameters (topics, alpha, eta, chunk size, etc.)
- Bengali-specific preprocessing: stemming, stopword removal, proper noun filtering
- Visualisations using Plotly, Bokeh, and Seaborn
- Report tab with structured interpretive summaries
- Drilldown views: document-topic and topic-term exploration
- Clean, responsive interface with dark mode and custom styling
- Downloadable outputs (CSV, TXT)
- Robust Documentations section that walks the user through topic modelling as a concept and how to anvay

## Requirements

- Python 3.9+
- Flask
- gensim
- nltk
- pandas
- matplotlib
- plotly
- bokeh
- pyLDAvis
- seaborn

A full list is available in `requirements.txt`.

## Running the App

```bash
pip install -r requirements.txt
python app.py
```

Then open `http://localhost:5000/` in your browser.

## Project Structure

- `app.py` — main Flask app
- `viz.py` — visualisation generation
- `templates/` — HTML templates
- `static/` — CSS and JS files
- `results/` — generated visualisation outputs

## License

anvay is released under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. Source code is available at:  
https://github.com/vinayakdasgupta/anvay
