import pandas as pd
import numpy as np
import re
import os
from Config import Config


def get_input_data() -> pd.DataFrame:
    """Load input CSV data from the configured path."""
    path = Config.DATA_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Data file not found at '{path}'. "
            "Please place your CSV file at that location."
        )
    df = pd.read_csv(path)
    print(f"[preprocess] Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def de_duplication(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows from the dataframe."""
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"[preprocess] De-duplication: removed {before - after} duplicate rows ({after} remaining)")
    return df


def noise_remover(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove noise from text columns:
    - Drop rows with null values in key columns
    - Strip extra whitespace
    - Remove special characters / non-ASCII from text fields
    """
    key_cols = [Config.TICKET_SUMMARY, Config.INTERACTION_CONTENT] + Config.TYPE_COLS
    # Keep only columns that actually exist in the dataframe
    key_cols = [c for c in key_cols if c in df.columns]

    before = len(df)
    df = df.dropna(subset=key_cols)
    after = len(df)
    print(f"[preprocess] Noise removal: dropped {before - after} rows with nulls ({after} remaining)")

    # Clean text in TICKET_SUMMARY and INTERACTION_CONTENT
    for col in [Config.TICKET_SUMMARY, Config.INTERACTION_CONTENT]:
        if col in df.columns:
            df[col] = df[col].astype(str).apply(_clean_text)

    df = df.reset_index(drop=True)
    return df


def _clean_text(text: str) -> str:
    """Helper: lowercase, strip HTML tags, collapse whitespace."""
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)          # remove HTML tags
    text = re.sub(r'[^a-z0-9\s]', ' ', text)      # keep only alphanumeric
    text = re.sub(r'\s+', ' ', text).strip()       # collapse whitespace
    return text


def translate_to_en(texts: list) -> list:
    """
    Translate texts to English.
    NOTE: For a real project, plug in a translation API (e.g. Google Translate).
    Here we return the texts unchanged as a safe stub — most CA datasets are
    already in English.
    """
    # Stub: return as-is
    return texts