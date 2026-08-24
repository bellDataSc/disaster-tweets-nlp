from __future__ import annotations

from pathlib import Path

import pytest

from disaster_tweets import data
from disaster_tweets.validation import ContractViolation

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_load_train_valid():
    df = data.load_train(FIXTURES_DIR / "train_valid.csv")
    assert list(df.columns) == ["id", "keyword", "location", "text", "target"]
    assert len(df) == 10


def test_load_train_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        data.load_train(FIXTURES_DIR / "does_not_exist.csv")


def test_load_train_invalid_target_raises_contract_violation():
    with pytest.raises(ContractViolation):
        data.load_train(FIXTURES_DIR / "train_invalid_target.csv")


def test_load_train_missing_column_raises():
    with pytest.raises(ContractViolation):
        data.load_train(FIXTURES_DIR / "train_missing_column.csv")


def test_load_train_duplicate_id_raises():
    with pytest.raises(ContractViolation):
        data.load_train(FIXTURES_DIR / "train_duplicate_id.csv")


def test_load_train_allows_null_keyword_and_location():
    df = data.load_train(FIXTURES_DIR / "train_valid.csv")
    assert df["keyword"].isnull().any()
    assert df["location"].isnull().any()


def test_get_conflicting_label_duplicates_detects_expected_rows():
    df = data.load_train(FIXTURES_DIR / "train_valid.csv", validate=False)
    conflicts = data.get_conflicting_label_duplicates(df)
    assert set(conflicts["text"]) == {"Forest fire near my house right now"}


def test_get_duplicate_text_groups_detects_expected_rows():
    df = data.load_train(FIXTURES_DIR / "train_valid.csv", validate=False)
    duplicates = data.get_duplicate_text_groups(df)
    assert len(duplicates) == 2


def test_load_test_valid():
    df = data.load_test(FIXTURES_DIR / "test_valid.csv")
    assert list(df.columns) == ["id", "keyword", "location", "text"]
    assert len(df) == 3


def test_load_sample_submission_valid():
    df = data.load_sample_submission(FIXTURES_DIR / "sample_submission_valid.csv")
    assert list(df.columns) == ["id", "target"]


def test_load_sample_submission_invalid_target_raises():
    with pytest.raises(ContractViolation):
        data.load_sample_submission(FIXTURES_DIR / "sample_submission_invalid.csv")


def test_load_afinn_lexicon_valid():
    df = data.load_afinn_lexicon(FIXTURES_DIR / "afinn_sample.txt")
    assert list(df.columns) == ["word", "score"]
    assert len(df) == 5


def test_load_afinn_lexicon_decodes_latin1_correctly(tmp_path):
    accented_path = tmp_path / "afinn_latin1.txt"
    accented_path.write_bytes("café\t1\n".encode("latin-1"))
    df = data.load_afinn_lexicon(accented_path, validate=False)
    assert "café" in set(df["word"])
    row = df[df["word"] == "café"].iloc[0]
    assert row["score"] == 1


def test_load_bing_lexicon_skips_comment_header():
    df = data.load_bing_lexicon(
        FIXTURES_DIR / "bing_positive_sample.txt",
        FIXTURES_DIR / "bing_negative_sample.txt",
    )
    assert set(df.columns) == {"word", "sentiment"}
    assert "happy" in set(df[df["sentiment"] == "positive"]["word"])
    assert "disaster" in set(df[df["sentiment"] == "negative"]["word"])


def test_load_nrc_lexicon_missing_file_raises_actionable_error():
    with pytest.raises(FileNotFoundError, match="manual download"):
        data.load_nrc_lexicon(FIXTURES_DIR / "does_not_exist.txt")


def test_load_nrc_lexicon_valid():
    df = data.load_nrc_lexicon(FIXTURES_DIR / "nrc_sample.txt")
    assert list(df.columns) == ["word", "emotion", "association"]
    assert len(df) == 8


def test_get_train_test_text_overlap():
    train_df = data.load_train(FIXTURES_DIR / "train_valid.csv", validate=False)
    test_df = data.load_test(FIXTURES_DIR / "test_valid.csv", validate=False)
    overlap = data.get_train_test_text_overlap(train_df, test_df)
    assert "Forest fire near my house right now" in set(overlap)
