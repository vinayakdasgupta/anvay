# Test Suite for Anvay

This directory contains the complete test suite for **anvay**, a Bengali topic modelling and visualisation tool.

These tests demonstrate:
- That the core pipeline and utilities function correctly
- That all visualisation modules produce valid HTML output
- That the Flask web interface handles user input and error cases gracefully

## Structure

| File | Purpose |
|------|---------|
| `test_utils.py` | Tests all utility functions (`utils.py`) — tokenisation, stemming, hyperparameters, etc. |
| `test_viz.py` | Tests all visualisation functions (`viz.py`) — ensures Plotly/HTML outputs render properly |
| `test_model_pipeline.py` | Tests end-to-end LDA training using `process_txt_files()` across different settings |
| `test_routes.py` | Tests key Flask routes (`/`, `/process`, `/about`, downloads, error pages) using `app.test_client()` |

## Running the Tests

To run all tests:

```bash
pytest

```
To run a specific file:

```bash
pytest tests/test_utils.py

```

## Notes

- All tests use minimal, synthetic Bengali text samples embedded directly in the test files.

- This ensures the suite is fast, self-contained, and does not rely on external corpora or large data files.

- Visualisation functions are tested by checking for valid Plotly-generated HTML snippets (presence of '<div>').

- The Flask app is tested internally via 'app.test_client()' — no live server or browser is required.
