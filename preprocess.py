# preprocess.py
# =============================================================================
# All preprocessing steps for the email classification pipeline.
# Functions called by main.py:
#   - get_input_data()    : load & merge both CSVs, rename Type cols -> y cols
#   - de_duplication()    : remove duplicate rows
#   - noise_remover()     : clean HTML entities, emails, anonymised tokens, etc.
#   - translate_to_en()   : translate non-English text to English
# =============================================================================

import re
import os
import numpy as np
import pandas as pd
from Config import Config

# ── File paths (adjust if your CSVs are in a subfolder) ──────────────────────
DATA_FILES = [
    os.path.join(os.path.dirname(__file__), 'AppGallery.csv'),
    os.path.join(os.path.dirname(__file__), 'Purchasing.csv'),
]

# =============================================================================
# 1. get_input_data
# =============================================================================

def get_input_data() -> pd.DataFrame:
    """
    Load all CSV files, concatenate them into one DataFrame, and rename
    the raw 'Type 1/2/3/4' columns to 'y1/y2/y3/y4' as expected by Config.

    Returns
    -------
    pd.DataFrame
        Combined, column-renamed DataFrame ready for preprocessing.
    """
    frames = []
    for path in DATA_FILES:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"[DataLoader] CSV file not found: {path}\n"
                "  -> Update DATA_FILES paths in preprocess.py to match your setup."
            )
        df = pd.read_csv(path, encoding='utf-8-sig', on_bad_lines='skip')
        print(f"[DataLoader] Loaded {len(df)} rows from '{os.path.basename(path)}'.")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    # ── Rename Type 1/2/3/4  →  y1/y2/y3/y4 ─────────────────────────────────
    rename_map = {
        'Type 1': 'y1',
        'Type 2': 'y2',
        'Type 3': 'y3',
        'Type 4': 'y4',
    }
    combined = combined.rename(columns=rename_map)

    # ── Drop unnamed trailing columns created by extra commas in CSV ──────────
    unnamed_cols = [c for c in combined.columns if str(c).startswith('Unnamed')]
    if unnamed_cols:
        combined = combined.drop(columns=unnamed_cols)

    print(f"[DataLoader] Total rows after merge: {len(combined)}.")
    return combined


# =============================================================================
# 2. de_duplication
# =============================================================================

def de_duplication(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove fully duplicate rows (all columns identical).
    Also removes rows where both the email text columns are null —
    these carry no signal for classification.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame  –  deduplicated
    """
    before = len(df)

    # Drop fully identical rows
    df = df.drop_duplicates()

    # Drop rows where BOTH text columns are empty (no useful signal)
    both_empty = (
        df[Config.TICKET_SUMMARY].isna() &
        df[Config.INTERACTION_CONTENT].isna()
    )
    df = df[~both_empty].copy()
    df = df.reset_index(drop=True)

    removed = before - len(df)
    print(f"[DeDuplication] Removed {removed} duplicate/empty rows. "
          f"{len(df)} rows remain.")
    return df


# =============================================================================
# 3. noise_remover
# =============================================================================

# Compiled regex patterns (compiled once for speed)
_RE_HTML_ENTITY   = re.compile(r'&[a-zA-Z]{2,6};')          # &amp; &lt; &gt; etc.
_RE_EMAIL         = re.compile(r'\S+@\S+\.\S+')              # email addresses
_RE_URL           = re.compile(r'https?://\S+|www\.\S+')     # URLs
_RE_ANON_TOKEN    = re.compile(r'\*+\([A-Z]+\)')             # *****(PER) *****(PHONE)
_RE_MASKED_EMAIL  = re.compile(r'[A-Za-z]+xxx+@\S+')        # Mxxxxx@xxxx.com
_RE_PHONE         = re.compile(r'\b\d[\d\s\-\(\)\.]{6,}\d\b')  # phone numbers
_RE_SPECIAL_CHARS = re.compile(r'[^a-zA-Z0-9\s\.,!?\'\"àáâãäåæçèéêëìíîïðñòóôõöùúûüý'
                                r'ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖÙÚÛÜÝ\u0400-\u04FF]')
_RE_WHITESPACE    = re.compile(r'\s{2,}')                    # multiple spaces


def _clean_text(text: str) -> str:
    """Apply all noise-removal patterns to a single string."""
    if not isinstance(text, str) or text.strip() == '':
        return ''

    text = _RE_HTML_ENTITY.sub(' ', text)     # &amp; -> space
    text = _RE_URL.sub(' ', text)             # remove URLs
    text = _RE_EMAIL.sub(' ', text)           # remove email addresses
    text = _RE_MASKED_EMAIL.sub(' ', text)    # remove masked emails (Mxxxxx@)
    text = _RE_ANON_TOKEN.sub(' ', text)      # remove *****(PER), *****(PHONE) etc.
    text = _RE_PHONE.sub(' ', text)           # remove phone numbers
    text = _RE_SPECIAL_CHARS.sub(' ', text)   # remove remaining special chars
    text = _RE_WHITESPACE.sub(' ', text)      # collapse whitespace
    return text.strip()


def noise_remover(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean both text columns in-place:
      - Strip HTML entities  (e.g. &amp; → ' ')
      - Remove URLs, email addresses, masked emails
      - Remove anonymised tokens: *****(PER), *****(PHONE), *****(LOC)
      - Remove phone numbers
      - Collapse extra whitespace
    Also fills remaining NaN values with empty strings so downstream
    operations never encounter NaN in the text columns.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame  –  cleaned
    """
    df = df.copy()

    for col in [Config.TICKET_SUMMARY, Config.INTERACTION_CONTENT]:
        df[col] = df[col].fillna('').astype(str)
        df[col] = df[col].apply(_clean_text)

    # Also clean the y1 column which contains &amp; entities
    if 'y1' in df.columns:
        df['y1'] = df['y1'].apply(lambda x: _RE_HTML_ENTITY.sub('&', str(x)).strip())

    print(f"[NoiseRemover] Cleaned text in '{Config.TICKET_SUMMARY}' "
          f"and '{Config.INTERACTION_CONTENT}'.")
    return df


# =============================================================================
# 4. translate_to_en
# =============================================================================

def _detect_language(text: str) -> str:
    """
    Lightweight language detector using Unicode character-range analysis.
    Returns an ISO-639-1 language code string, or 'en' if uncertain.

    Detected ranges:
        Cyrillic  (Russian, Ukrainian, Bulgarian, …)  → 'ru'
        Greek                                          → 'el'
        Arabic                                         → 'ar'
        CJK (Chinese / Japanese / Korean)              → 'zh'
        Otherwise                                      → 'en'  (assume Latin / English)

    Note: This intentionally does NOT distinguish between Latin-script
    languages (German, French, Italian, Portuguese, etc.) — those cases
    are handled by the translation library which auto-detects the source
    language before translating.
    """
    if not isinstance(text, str) or len(text.strip()) < 3:
        return 'en'

    counts = {
        'cyrillic': 0,
        'greek':    0,
        'arabic':   0,
        'cjk':      0,
        'latin':    0,
    }
    for ch in text:
        cp = ord(ch)
        if 0x0400 <= cp <= 0x04FF:           counts['cyrillic'] += 1
        elif 0x0370 <= cp <= 0x03FF:         counts['greek']    += 1
        elif 0x0600 <= cp <= 0x06FF:         counts['arabic']   += 1
        elif (0x4E00 <= cp <= 0x9FFF or
              0x3040 <= cp <= 0x30FF or
              0xAC00 <= cp <= 0xD7FF):       counts['cjk']      += 1
        elif (0x0041 <= cp <= 0x007A or
              0x00C0 <= cp <= 0x024F):       counts['latin']    += 1

    total = max(sum(counts.values()), 1)
    dominant = max(counts, key=counts.get)
    dominant_ratio = counts[dominant] / total

    if dominant_ratio < 0.15:
        return 'en'   # mixed / mostly punctuation → treat as English

    if dominant == 'cyrillic': return 'ru'
    if dominant == 'greek':    return 'el'
    if dominant == 'arabic':   return 'ar'
    if dominant == 'cjk':      return 'zh'
    # Latin-script: could be de/fr/it/pt/es/en — return 'latin'
    # (translation library will handle the exact language detection)
    return 'latin'


def _try_translate(text: str) -> str:
    """
    Attempt translation to English using available libraries.
    Tries deep_translator first (most reliable), then googletrans as fallback.
    If neither is installed or the network is unavailable, returns text unchanged.

    Install with:
        pip install deep-translator
    """
    # ── Attempt 1: deep_translator ────────────────────────────────────────────
    try:
        from deep_translator import GoogleTranslator
        translated = GoogleTranslator(source='auto', target='en').translate(text)
        return translated if translated else text
    except Exception:
        pass

    # ── Attempt 2: googletrans ────────────────────────────────────────────────
    try:
        from googletrans import Translator
        translator = Translator()
        result = translator.translate(text, dest='en')
        return result.text if result and result.text else text
    except Exception:
        pass

    # ── Fallback: return original if no library available ─────────────────────
    return text


def translate_to_en(texts: list) -> list:
    """
    Translate a list of strings to English.

    Strategy:
      1. Detect language of each string.
      2. If language appears to be non-Latin (Cyrillic, Arabic, etc.)
         OR if the text contains non-ASCII characters suggesting a foreign
         Latin-script language, call the translation API.
      3. Already-English strings are returned unchanged (no API call).

    Parameters
    ----------
    texts : list of str

    Returns
    -------
    list of str  –  translated (or original if already English / translation unavailable)
    """
    translated_count = 0
    result = []

    for text in texts:
        if not isinstance(text, str) or text.strip() == '':
            result.append(text)
            continue

        lang = _detect_language(text)

        # Check for non-ASCII characters even in 'latin' category
        # (catches German umlauts, French accents, Italian, Portuguese, Spanish)
        has_non_ascii = any(ord(c) > 127 for c in text)

        if lang != 'en' or has_non_ascii:
            translated = _try_translate(text)
            if translated != text:
                translated_count += 1
            result.append(translated)
        else:
            result.append(text)

    print(f"[Translate] Processed {len(texts)} strings. "
          f"Translated {translated_count}. "
          f"({'deep-translator/googletrans required — pip install deep-translator' if translated_count == 0 else 'OK'})")
    return result
