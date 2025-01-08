# LDA Web Application for Text Corpus Analysis

This Flask-based web application allows users to upload `.txt` files, processes the content using Latent Dirichlet Allocation (LDA), and generates an interactive visualization of the topics present in the text corpus. The application supports multiple file uploads and generates a visualization using `pyLDAvis`.

## Features
- Upload multiple `.txt` files containing text data (e.g., Bengali or English text).
- Tokenization, stopword removal, and bigram creation are applied to the text data.
- LDA model is built on the processed text data to discover topics.
- Interactive visualization of topics using `pyLDAvis`.
- Easy-to-use web interface for file upload and visualization download.

## Requirements
This project requires Python 3.x. You can install the required dependencies by running the following command:

```bash
pip install -r requirements.txt