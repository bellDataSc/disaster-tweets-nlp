from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from pandas.errors import EmptyDataError

from disaster_tweets.config import get_config
from disaster_tweets.validation import (
    validate_lexicon_frame,
    validate_sample_submission_frame,
    validate_test_frame,
    validate_train_frame,
)


def _resolve_raw_path(path: str | Path | None, filename: str) -> Path:
    if path is not None:
        return Path(path)
    return get_config().paths.raw_data_dir / filename


def _read_csv_safely(path: Path, **kwargs: Any) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path}. Run 'python scripts/download_data.py' "
            f"or place the file manually in the expected location."
        )
    try:
        df = pd.read_csv(path, **kwargs)
    except EmptyDataError as exc:
        raise ValueError(f"File is empty: {path}") from exc
    return pd.DataFrame(df)


def load_train(path: str | Path | None = None, validate: bool = True) -> pd.DataFrame:
    resolved = _resolve_raw_path(path, "train.csv")
    df = _read_csv_safely(resolved, encoding="utf-8-sig")
    if validate:
        validate_train_frame(df)
    return df


def load_test(path: str | Path | None = None, validate: bool = True) -> pd.DataFrame:
    resolved = _resolve_raw_path(path, "test.csv")
    df = _read_csv_safely(resolved, encoding="utf-8-sig")
    if validate:
        validate_test_frame(df)
    return df


def load_sample_submission(path: str | Path | None = None, validate: bool = True) -> pd.DataFrame:
    resolved = _resolve_raw_path(path, "sample_submission.csv")
    df = _read_csv_safely(resolved, encoding="utf-8-sig")
    if validate:
        validate_sample_submission_frame(df)
    return df


def load_afinn_lexicon(path: str | Path | None = None, validate: bool = True) -> pd.DataFrame:
    resolved = _resolve_raw_path(path, "lexicons/afinn_en_165.txt")
    df = _read_csv_safely(
        resolved,
        sep="\t",
        header=None,
        names=["word", "score"],
        encoding="latin-1",
    )
    if validate:
        validate_lexicon_frame(df, kind="afinn")
    return df


def _read_bing_wordlist(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path}. Run 'python scripts/download_data.py' "
            f"or place the file manually in the expected location."
        )
    return pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["word"],
        comment=";",
        skip_blank_lines=True,
        encoding="latin-1",
    )


def load_bing_lexicon(
    positive_path: str | Path | None = None,
    negative_path: str | Path | None = None,
    validate: bool = True,
) -> pd.DataFrame:
    resolved_positive = _resolve_raw_path(positive_path, "lexicons/bing_positive.txt")
    resolved_negative = _resolve_raw_path(negative_path, "lexicons/bing_negative.txt")
    positive_df = _read_bing_wordlist(resolved_positive)
    positive_df["sentiment"] = "positive"
    negative_df = _read_bing_wordlist(resolved_negative)
    negative_df["sentiment"] = "negative"
    df = pd.concat([positive_df, negative_df], ignore_index=True)
    if validate:
        validate_lexicon_frame(df, kind="bing")
    return df


def load_nrc_lexicon(path: str | Path | None = None, validate: bool = True) -> pd.DataFrame:
    resolved = _resolve_raw_path(path, "lexicons/nrc_emotion_lexicon.txt")
    if not resolved.exists():
        raise FileNotFoundError(
            f"NRC lexicon not found at {resolved}. The NRC Emotion Lexicon requires "
            f"manual download from https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm "
            f"and must be placed at that path (file "
            f"'NRC-Emotion-Lexicon-Wordlevel-v0.92.txt' renamed to 'nrc_emotion_lexicon.txt')."
        )
    df = _read_csv_safely(
        resolved,
        sep="\t",
        header=None,
        names=["word", "emotion", "association"],
        encoding="utf-8",
    )
    if validate:
        validate_lexicon_frame(df, kind="nrc")
    return df


def get_duplicate_text_groups(df: pd.DataFrame) -> pd.DataFrame:
    counts = df.groupby("text").size()
    duplicated_texts = counts[counts > 1].index
    return df[df["text"].isin(duplicated_texts)].sort_values("text")


def get_conflicting_label_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    label_counts = df.groupby("text")["target"].nunique()
    conflicting_texts = label_counts[label_counts > 1].index
    return df[df["text"].isin(conflicting_texts)].sort_values("text")


def get_train_test_text_overlap(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.Series:
    overlap = set(train_df["text"]) & set(test_df["text"])
    return pd.Series(sorted(overlap), name="text")
