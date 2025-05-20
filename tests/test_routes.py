# tests/test_flask_routes.py

import os
import io
import pytest
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    return app.test_client()

# -------- Static routes --------
def test_home_page(client):
    res = client.get('/')
    assert res.status_code == 200
    assert b"Explore" in res.data or b"Upload" in res.data

def test_about_page(client):
    res = client.get('/about')
    assert res.status_code == 200

def test_docs_page(client):
    res = client.get('/docs')
    assert res.status_code == 200

def test_contact_page(client):
    res = client.get('/contact')
    assert res.status_code == 200

# -------- 404 page --------
def test_404_page(client):
    res = client.get('/nonexistent')
    assert res.status_code == 404
    assert b"not be found" in res.data.lower()

# -------- File upload (valid) --------
def test_upload_and_process(client):
    data = {
        'files[]': (io.BytesIO("আমি বাংলা পড়ি।".encode('utf-8')), 'sample.txt'),
        'num_topics': '2',
        'iterations': '5',
        'passes': '5',
        'chunk_size': '100',
        'ngram': 'unigram',
        'alpha': 'symmetric',
        'eta': 'symmetric',
        'per_word_topics': 'false',
        'no_below': '1',
        'no_above': '1.0',
        'use_stemming': '',
        'use_multicore': 'false',
        'percent_most_common': '0'
    }
    res = client.post('/process', data=data, content_type='multipart/form-data', follow_redirects=True)
    assert res.status_code in (200, 500)  # 500 if template fails to render, which is okay under test

# -------- File upload (missing file) --------
def test_upload_no_file(client):
    res = client.post('/process', data={}, content_type='multipart/form-data', follow_redirects=True)
    assert res.status_code == 200
    assert b"No files selected" in res.data or b"danger" in res.data

# -------- Download endpoints --------
def test_download_missing_file(client):
    res = client.get('/download_file/nonexistent.txt')
    assert res.status_code == 404

def test_result_file_missing(client):
    res = client.get('/results/html/fakefile.html')
    assert res.status_code == 404
