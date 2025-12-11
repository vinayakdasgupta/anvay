## v1.1.1 — 2025-12-11

### Improvements
- Added top-word hover tooltips across all visualisations to improve topic interpretability.
- Standardised colour mapping so all topics use a consistent global colour scheme.
- Updated documentation to state that the Heatmap and Bar Chart visualise the same topic–word weight matrix.
- Reduced the number of terms displayed in selected visualisations to avoid tick-label suppression; added hover information where relevant (e.g., topic distribution x-axis).
- Clarified the axis label in the Topic Evolution plot (now explicitly tied to document upload order); added filename to hover information.
- Enhanced hierarchical clustering with BERTopic-style merged-cluster keyword summaries on hover.
- Unified visual style across plots using the Roboto / Noto Bengali font stack; reduced Plotly margins for cleaner layout.
- Added a loading spinner to indicate processing during analysis.

### Notes
- Hovering on Plotly legends is unfortunately not supported; tooltips are therefore provided directly on the plots.


## [1.1.0] – 2025-04

### Added

- Full walkthrough section with sample corpus and model settings
- Seven new visualisations (scatterplot, heatmap, pie chart, etc.)
- Sentence-level topic confidence detection
- Toggleable Bengali/English examples in documentation
- Glossary and parameter tooltips

### Changed

- Unified styling across docs and results
- Visualisation orientation detection for modal views
- Topic reports now use real filenames

### Fixed

- Tokenisation edge cases with malformed Bengali characters
- Layout bugs in Report tab accordion

### Removed

- λ-scored top terms (no longer shown)

### Notes

- Expect variation across identical runs due to LDA randomness
