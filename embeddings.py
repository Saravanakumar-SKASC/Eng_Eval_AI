# embeddings.py
# =============================================================================
# Converts the cleaned email text into numeric feature vectors (TF-IDF).
# Function called by main.py:
#   - get_tfidf_embd(df)  →  returns sparse TF-IDF matrix X
# =============================================================================

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from Config import Config

# ── Module-level vectorizer (fit once, reuse for any future transform calls) ─
_vectorizer: TfidfVectorizer = None


def _combine_text(df: pd.DataFrame) -> pd.Series:
    """
    Combine 'Ticket Summary' and 'Interaction content' into one text field.

    The Ticket Summary is a short subject line; the Interaction content is
    the full email body. Concatenating them (summary first, then body) gives
    the model access to both signals within a single TF-IDF representation.

    A space is used as separator. NaN values are already filled to ''
    by the noise_remover step, but we guard here with fillna just in case.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.Series  –  one combined string per email row
    """
    summary = df[Config.TICKET_SUMMARY].fillna('').astype(str)
    content = df[Config.INTERACTION_CONTENT].fillna('').astype(str)
    combined = summary + ' ' + content
    return combined.str.strip()


def get_tfidf_embd(df: pd.DataFrame) -> np.ndarray:
    """
    Fit a TF-IDF vectorizer on the combined email text and return the
    resulting feature matrix.

    Preprocessing steps inside TF-IDF:
      - Tokenisation on whitespace + punctuation
      - Lowercasing
      - Remove English stop words
      - 1-gram and 2-gram features  (captures phrases like "can't install")
      - Sublinear TF scaling        (log(1+tf) — reduces dominance of very
                                     frequent terms)
      - Max 5000 features           (keeps memory manageable on small data)
      - min_df=2                    (ignore terms appearing in only 1 document)

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame (after preprocess steps). Must contain
        Config.TICKET_SUMMARY and Config.INTERACTION_CONTENT columns.

    Returns
    -------
    np.ndarray  –  dense numpy array of shape (n_samples, n_features)
        Returned as a dense array so it is directly compatible with the
        Data() constructor and sklearn classifiers expecting a 2-D array.
    """
    global _vectorizer

    texts = _combine_text(df)

    _vectorizer = TfidfVectorizer(
        analyzer='word',
        tokenizer=None,          # default whitespace + punct tokeniser
        preprocessor=None,       # we already cleaned text in noise_remover
        stop_words='english',
        lowercase=True,
        ngram_range=(1, 2),      # unigrams + bigrams
        max_features=5000,
        min_df=2,                # ignore very rare terms
        sublinear_tf=True,       # log(1+tf) scaling
    )

    X_sparse = _vectorizer.fit_transform(texts)
    X = X_sparse.toarray()       # convert to dense numpy array

    print(f"[Embeddings] TF-IDF matrix: {X.shape[0]} samples × {X.shape[1]} features.")
    print(f"[Embeddings] Vocabulary size: {len(_vectorizer.vocabulary_)}.")
    return X


def transform_new(texts: list) -> np.ndarray:
    """
    Transform new (unseen) texts using the already-fitted TF-IDF vectorizer.
    Useful for inference / predicting on new emails after training.

    Parameters
    ----------
    texts : list of str

    Returns
    -------
    np.ndarray  –  TF-IDF feature matrix for the new texts

    Raises
    ------
    RuntimeError  –  if get_tfidf_embd() has not been called yet
    """
    if _vectorizer is None:
        raise RuntimeError(
            "[Embeddings] Vectorizer not fitted. "
            "Call get_tfidf_embd(df) on training data first."
        )
    X_sparse = _vectorizer.transform(texts)
    return X_sparse.toarray()
