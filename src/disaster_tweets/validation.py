from __future__ import annotations

from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, field_validator


class ContractViolation(Exception):
    def __init__(self, violations: list[str]) -> None:
        self.violations = violations
        super().__init__("; ".join(violations))


class DisasterTweetTrainRow(BaseModel):
    model_config = ConfigDict(strict=False)

    id: int
    keyword: str | None = None
    location: str | None = None
    text: str
    target: int

    @field_validator("target")
    @classmethod
    def target_must_be_binary(cls, value: int) -> int:
        if value not in (0, 1):
            raise ValueError(f"target must be 0 or 1, got {value}")
        return value


class DisasterTweetTestRow(BaseModel):
    model_config = ConfigDict(strict=False)

    id: int
    keyword: str | None = None
    location: str | None = None
    text: str


class SampleSubmissionRow(BaseModel):
    model_config = ConfigDict(strict=False)

    id: int
    target: int

    @field_validator("target")
    @classmethod
    def target_must_be_binary(cls, value: int) -> int:
        if value not in (0, 1):
            raise ValueError(f"target must be 0 or 1, got {value}")
        return value


class AfinnRow(BaseModel):
    model_config = ConfigDict(strict=False)

    word: str
    score: int


class BingRow(BaseModel):
    model_config = ConfigDict(strict=False)

    word: str
    sentiment: Literal["positive", "negative"]


class NrcRow(BaseModel):
    model_config = ConfigDict(strict=False)

    word: str
    emotion: str
    association: int


def _require_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    missing = [column for column in columns if column not in df.columns]
    return [f"missing required column: {column}" for column in missing]


def _check_unique(df: pd.DataFrame, column: str) -> list[str]:
    if column not in df.columns:
        return []
    duplicated = df[df[column].duplicated()][column].tolist()
    if duplicated:
        return [f"column '{column}' contains duplicate values: {duplicated[:5]}"]
    return []


def _check_no_nulls(df: pd.DataFrame, column: str) -> list[str]:
    if column not in df.columns:
        return []
    if df[column].isnull().any():
        return [f"column '{column}' contains null values"]
    return []


def validate_train_frame(df: pd.DataFrame) -> None:
    violations = _require_columns(df, ["id", "keyword", "location", "text", "target"])
    violations += _check_unique(df, "id")
    violations += _check_no_nulls(df, "id")
    violations += _check_no_nulls(df, "text")
    violations += _check_no_nulls(df, "target")
    if "target" in df.columns:
        invalid_targets = df[~df["target"].isin([0, 1])]
        if not invalid_targets.empty:
            bad_values = invalid_targets["target"].unique().tolist()
            violations.append(f"target column has values outside {{0, 1}}: {bad_values}")
    if violations:
        raise ContractViolation(violations)


def validate_test_frame(df: pd.DataFrame) -> None:
    violations = _require_columns(df, ["id", "keyword", "location", "text"])
    violations += _check_unique(df, "id")
    violations += _check_no_nulls(df, "id")
    violations += _check_no_nulls(df, "text")
    if violations:
        raise ContractViolation(violations)


def validate_sample_submission_frame(df: pd.DataFrame) -> None:
    violations = _require_columns(df, ["id", "target"])
    violations += _check_unique(df, "id")
    if "target" in df.columns:
        invalid_targets = df[~df["target"].isin([0, 1])]
        if not invalid_targets.empty:
            bad_values = invalid_targets["target"].unique().tolist()
            violations.append(f"target column has values outside {{0, 1}}: {bad_values}")
    if violations:
        raise ContractViolation(violations)


def validate_lexicon_frame(df: pd.DataFrame, kind: Literal["afinn", "bing", "nrc"]) -> None:
    required = {
        "afinn": ["word", "score"],
        "bing": ["word", "sentiment"],
        "nrc": ["word", "emotion", "association"],
    }[kind]
    violations = _require_columns(df, required)
    violations += _check_no_nulls(df, "word")
    if violations:
        raise ContractViolation(violations)
