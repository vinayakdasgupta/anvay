from preprocessing.bn import preprocess_bn


def preprocess_documents(
    file_paths,
    remove_stopwords,
    custom_stopwords,
    use_stemming,
    normalisation_order,
    percent,
    ngram,
    language="bn"
):
    if language == "bn":
        return preprocess_bn(
            file_paths,
            remove_stopwords,
            custom_stopwords,
            use_stemming,
            normalisation_order,
            percent,
            ngram
        )
    else:
        raise ValueError(f"Unsupported language: {language}")
