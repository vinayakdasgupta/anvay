# tests/test_model_pipeline.py

import os
import pytest
import tempfile
import shutil
from app import process_txt_files

# Sample Bengali documents for full pipeline testing
SAMPLE_TEXTS = [
    "আমি বাংলা লিখছি। সে বাংলা পড়ে।",
    "নদী বয়ে যায়। পাখি উড়ে যায়।",
    "বই পড়া ভালো অভ্যাস।"
]

@pytest.fixture(scope="module")
def temp_bengali_txt_files():
    temp_dir = tempfile.mkdtemp()
    file_paths = []
    for i, text in enumerate(SAMPLE_TEXTS):
        path = os.path.join(temp_dir, f"doc_{i}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        file_paths.append(path)
    yield file_paths
    shutil.rmtree(temp_dir)

def test_basic_pipeline_runs(temp_bengali_txt_files):
    result = process_txt_files(
        file_paths=temp_bengali_txt_files,
        num_topics=2,
        iterations=5,
        passes=5,
        minimum_probability=0.01,
        chunk_size=100,
        ngram="unigram",
        alpha="symmetric",
        eta="symmetric",
        per_word_topics=False,
        no_below=1,
        no_above=1.0,
        use_stemming=False,
        use_multicore=False,
        percent=0,
        remove_stopwords=False
    )
    txt_path, csv_path, lda_model, corpus, id2word, tokens, overview, *_ = result
    assert os.path.exists(txt_path)
    assert os.path.exists(csv_path)
    assert isinstance(tokens, list)
    assert isinstance(lda_model.num_topics, int)
    assert isinstance(corpus, list)
    assert isinstance(id2word.token2id, dict)

def test_pipeline_with_bigrams(temp_bengali_txt_files):
    result = process_txt_files(
        file_paths=temp_bengali_txt_files,
        num_topics=2,
        iterations=3,
        passes=3,
        minimum_probability=0.01,
        chunk_size=100,
        ngram="bigram",
        alpha="symmetric",
        eta="symmetric",
        per_word_topics=False,
        no_below=1,
        no_above=1.0,
        use_stemming=True,
        use_multicore=False,
        percent=0,
        remove_stopwords=True
    )
    lda_model, corpus, _, tokens, *_ = result[2:7]
    assert lda_model.num_topics == 2
    assert isinstance(tokens[0], list)

def test_pipeline_invalid_filter_threshold(temp_bengali_txt_files):
    with pytest.raises(ValueError, match="Dictionary is empty after filtering"):
        process_txt_files(
            file_paths=temp_bengali_txt_files,
            num_topics=2,
            iterations=3,
            passes=3,
            minimum_probability=0.01,
            chunk_size=100,
            ngram="unigram",
            alpha="symmetric",
            eta="symmetric",
            per_word_topics=False,
            no_below=100,  # too high — triggers empty dict
            no_above=0.1,
            use_stemming=False,
            use_multicore=False,
            percent=0,
            remove_stopwords=False
        )
