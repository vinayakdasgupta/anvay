from dataclasses import dataclass
from typing import Union, Optional

from utils import parse_hyperparam


@dataclass(frozen=True)
class AnalysisConfig:
    # -----------------------------
    # Core LDA
    # -----------------------------
    num_topics: int
    iterations: int
    passes: int
    chunk_size: int
    alpha: Union[str, float]
    eta: Union[str, float]
    per_word_topics: bool
    minimum_probability: float

    # -----------------------------
    # Vocabulary filtering
    # -----------------------------
    no_below: int
    no_above: float
    percent_most_common: float

    # -----------------------------
    # Preprocessing (intent only)
    # -----------------------------
    ngram: str
    normalisation: Optional[str]        # "lemma" | "stem" | "none" | "auto"
    normalisation_order: str            # passed through (BN-only semantics)
    remove_stopwords: bool

    # -----------------------------
    # Execution
    # -----------------------------
    use_multicore: bool

    # -----------------------------
    # Language selection
    # -----------------------------
    language: str


def build_analysis_config(form) -> AnalysisConfig:
    """
    Build analysis configuration from request.form.

    NOTE:
    - No language policy lives here.
    - No normaliser resolution happens here.
    """

    return AnalysisConfig(
        # Core LDA
        num_topics=int(form.get("num_topics", 10)),
        iterations=int(form.get("iterations", 40)),
        passes=int(form.get("passes", 10)),
        chunk_size=int(form.get("chunk_size", 200)),
        alpha=parse_hyperparam(form.get("alpha", "symmetric")),
        eta=parse_hyperparam(form.get("eta", "symmetric")),
        per_word_topics=form.get("per_word_topics", "false").lower() == "true",
        minimum_probability=0.1,

        # Vocabulary filtering
        no_below=int(form.get("no_below", 1)),
        no_above=float(form.get("no_above", 0.9)),
        percent_most_common=float(form.get("percent_most_common", 0)),

        # Preprocessing intent
        ngram=form.get("ngram", "unigram"),
        normalisation=form.get("normalisation", "auto"),
        normalisation_order=form.get("normalisation_order", "stem_first"),
        remove_stopwords=form.get("remove_nltk_stopwords", "true").lower() == "true",

        # Execution
        use_multicore=form.get("use_multicore", "false").lower() == "true",

        # Language
        language=form.get("language", "bn"),
    )
