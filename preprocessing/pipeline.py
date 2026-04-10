# Import strategy: try package-style imports first (when running from project root
# with preprocessing/ as a subpackage), then fall back to flat/sibling imports
# (when files are in the same directory, e.g. during development or testing).
try:
    from preprocessing.bn import preprocess_bn
    from preprocessing.en import preprocess_en
    from preprocessing.hi import preprocess_hi
    from preprocessing.ta import preprocess_ta
    from preprocessing.normalisation import (
        resolve_normaliser,
        resolve_default_normaliser,
    )
except ImportError:
    from bn import preprocess_bn
    from en import preprocess_en
    from hi import preprocess_hi
    from ta import preprocess_ta
    from normalisation import (
        resolve_normaliser,
        resolve_default_normaliser,
    )


def preprocess_documents(
    file_paths,
    remove_stopwords,
    custom_stopwords,
    normalisation,          # user request: "lemma" | "stem" | "none" | None
    normalisation_order,
    percent,
    ngram,
    language="bn"
):
    """
    Preprocess documents with language-specific logic and
    resolved lexical normalisation.
    """

    # --- resolve normaliser centrally ---
    if normalisation in (None, "auto"):
        normaliser = resolve_default_normaliser(language)
    else:
        normaliser = resolve_normaliser(language, normalisation)

    # --- dispatch to language-specific preprocessors ---
    if language == "bn":
        return preprocess_bn(
            file_paths=file_paths,
            remove_stopwords=remove_stopwords,
            custom_stopwords=custom_stopwords,
            normaliser=normaliser,                 # ← NEW
            normalisation_order=normalisation_order,
            percent=percent,
            ngram=ngram
        )

    elif language == "en":
        return preprocess_en(
            file_paths=file_paths,
            remove_stopwords=remove_stopwords,
            custom_stopwords=custom_stopwords,
            normaliser=normaliser,                 # ← NEW
            percent=percent,
            ngram=ngram
        )
    elif language == "hi":
        return preprocess_hi(
            file_paths=file_paths,
            remove_stopwords=remove_stopwords,
            custom_stopwords=custom_stopwords,
            normaliser=normaliser,
            percent=percent,
            ngram=ngram
        )
    elif language == "ta":
        return preprocess_ta(
            file_paths=file_paths,
            remove_stopwords=remove_stopwords,
            custom_stopwords=custom_stopwords,
            normaliser=normaliser,
            percent=percent,
            ngram=ngram
        )

    else:
        raise ValueError(f"Unsupported language: {language}")
