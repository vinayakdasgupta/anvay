# anvay v1.1.1 — Release Notes

This update focuses on improving topic interpretability, visual consistency, and user guidance throughout the platform. All changes below are based directly on reviewer feedback and user testing.

## 🧠 Topic Interpretability Enhancements
- **Top words now appear on hover in every visualisation**, allowing users to interpret topics without switching back and forth between tabs.
- **A unified topic colour scheme** is now applied globally.
- (Note: Plotly does not support hover tooltips on legends, so this interaction is implemented directly on plot elements.)

## 📊 Visualisation Clarity
- Added a clear statement in the documentation noting that the **Heatmap and Bar Chart represent the same topic–word weight matrix**.
- Reduced the number of displayed terms in certain charts to prevent Plotly from hiding tick labels.
- Added supplemental hover information to axes where appropriate (e.g., topic distribution x-axis).
- In the **Topic Evolution** plot, clarified that the x-axis reflects **document upload order**, and added filenames to hover data.

## 🌳 Improved Topic Clustering
- Hierarchical clustering now includes **BERTopic-style keyword aggregation for merged clusters**, making cluster interpretation significantly easier.

## 🎨 Visual Streamlining
- All visualisations now use the **Roboto / Noto Bengali font stack** for consistency with the rest of the site.
- Reduced Plotly figure margins for a cleaner and more integrated layout.

## ⚙️ UI/UX Improvements
- Added a **loading spinner** to the analysis workflow so that users receive clear feedback during processing.

## 🙏 Acknowledgment
Many thanks for the suggestions, @x-tabdeveloping — these changes make the visualisations far more intuitive and greatly enhance the overall user experience.


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
