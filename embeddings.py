import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from Config import Config


def get_tfidf_embd(df: pd.DataFrame) -> np.ndarray:
    """
    Combine Ticket Summary and Interaction Content into one text field,
    then produce a TF-IDF embedding matrix.

    Returns
    -------
    X : np.ndarray of shape (n_samples, n_features)
    """
    # Combine the two text columns into a single feature
    combined = (
        df[Config.TICKET_SUMMARY].astype(str)
        + " "
        + df[Config.INTERACTION_CONTENT].astype(str)
    )

    vectorizer = TfidfVectorizer(
        max_features=5000,   # keep top 5 000 terms
        sublinear_tf=True,   # apply log normalization to TF
        min_df=2,            # ignore terms appearing in fewer than 2 docs
        ngram_range=(1, 2),  # unigrams + bigrams
        stop_words='english'
    )

    X = vectorizer.fit_transform(combined).toarray()
    print(f"[embeddings] TF-IDF matrix shape: {X.shape}")
    return X
