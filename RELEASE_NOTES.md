# Release Notes – anvay v1.0.0
April 2025

This is the first public release of **anvay**, a Bengali LDA topic modelling tool designed for web-based exploratory use.

## Highlights

- Full pipeline: upload, model, visualise, interpret
- Parameter configuration interface for LDA (passes, iterations, alpha, etc.)
- Bengali-specific preprocessing: stopwords, stemming, noise filtering
- Four visualisation types: PyLDAvis, Bokeh, Plotly, Seaborn
- Structured Report tab with topic-document mapping, training overview, and top terms
- Contextual document preview with filename tracking
- "Low Confidence" topic warnings for spurious results
- Clean, mobile-responsive interface with light/dark mode
- Downloadable CSV and TXT outputs
- Documentation and sample corpus included

## Known Issues

- Some malformed tokens may still appear if UTF-8 encoding is not preserved
- Contextual sentence previews rely on approximate token overlap and may miss nuance
- Heatmap visualisation may not render correctly on small screens

## Coming 

- Support for more Indic languages
- Deployment via Gunicorn and Docker
- Server-side queuing for high load

Feedback and issues can be raised at:  
https://github.com/vinayakdasgupta/anvay/issues
