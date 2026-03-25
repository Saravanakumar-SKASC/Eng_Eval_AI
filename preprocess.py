import re
import os
import numpy as np
import pandas as pd
from Config import Config

DATA_FILES = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'AppGallery.csv'),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'Purchasing.csv'),
]

def get_input_data() -> pd.DataFrame:
    frames = []
    for path in DATA_FILES:
        df = pd.read_csv(path, encoding='utf-8-sig', on_bad_lines='skip')
        print(f"[DataLoader] Loaded {len(df)} rows from '{os.path.basename(path)}'.")
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.rename(columns={
        'Type 1': 'y1', 'Type 2': 'y2', 'Type 3': 'y3', 'Type 4': 'y4'
    })
    unnamed = [c for c in combined.columns if str(c).startswith('Unnamed')]
    if unnamed:
        combined = combined.drop(columns=unnamed)
    print(f"[DataLoader] Total rows: {len(combined)}.")
    return combined

def de_duplication(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    both_empty = df[Config.TICKET_SUMMARY].isna() & df[Config.INTERACTION_CONTENT].isna()
    df = df[~both_empty].copy().reset_index(drop=True)
    print(f"[DeDuplication] Removed {before - len(df)} rows. {len(df)} remain.")
    return df

_RE_HTML   = re.compile(r'&[a-zA-Z]{2,6};')
_RE_EMAIL  = re.compile(r'\S+@\S+\.\S+')
_RE_URL    = re.compile(r'https?://\S+|www\.\S+')
_RE_ANON   = re.compile(r'\*+\([A-Z]+\)')
_RE_MASKED = re.compile(r'[A-Za-z]+xxx+@\S+')
_RE_PHONE  = re.compile(r'\b\d[\d\s\-\(\)\.]{6,}\d\b')
_RE_SPACE  = re.compile(r'\s{2,}')

def _clean(text: str) -> str:
    if not isinstance(text, str):
        return ''
    text = _RE_HTML.sub(' ', text)
    text = _RE_URL.sub(' ', text)
    text = _RE_EMAIL.sub(' ', text)
    text = _RE_MASKED.sub(' ', text)
    text = _RE_ANON.sub(' ', text)
    text = _RE_PHONE.sub(' ', text)
    text = _RE_SPACE.sub(' ', text)
    return text.strip()

def noise_remover(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in [Config.TICKET_SUMMARY, Config.INTERACTION_CONTENT]:
        df[col] = df[col].fillna('').astype(str).apply(_clean)
    if 'y1' in df.columns:
        df['y1'] = df['y1'].apply(lambda x: _RE_HTML.sub('&', str(x)).strip())
    print(f"[NoiseRemover] Cleaned text columns.")
    return df

def translate_to_en(texts: list) -> list:
    result = []
    for text in texts:
        if not isinstance(text, str) or text.strip() == '':
            result.append(text)
            continue
        has_non_ascii = any(ord(c) > 127 for c in text)
        if has_non_ascii:
            try:
                from deep_translator import GoogleTranslator
                translated = GoogleTranslator(source='auto', target='en').translate(text)
                result.append(translated if translated else text)
                continue
            except Exception:
                pass
        result.append(text)
    print(f"[Translate] Processed {len(texts)} strings.")
    return result
