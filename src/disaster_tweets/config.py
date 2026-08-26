from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel


class Paths(BaseModel):
    data_dir: Path
    raw_data_dir: Path
    interim_data_dir: Path
    processed_data_dir: Path
    artifacts_dir: Path
    reports_dir: Path


class AppConfig(BaseModel):
    project_root: Path
    paths: Paths
    random_seed: int = 42
    default_config: Path
    kaggle_competition: str = "nlp-getting-started"
    log_level: str = "INFO"
    app_env: str = "development"


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path(__file__)).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError(
        "Could not locate project root: no pyproject.toml found in any parent directory"
    )


def load_env(env_file: Path | None = None) -> None:
    root = find_project_root()
    target = env_file or (root / ".env")
    if target.exists():
        load_dotenv(target)


def load_yaml_config(path: str | Path) -> dict:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Config file not found: {resolved}")
    with open(resolved, encoding="utf-8") as handle:
        content = yaml.safe_load(handle)
    if not isinstance(content, dict):
        raise ValueError(f"Config file did not parse to a mapping: {resolved}")
    return content


def _resolve_path(root: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else root / candidate


def build_config() -> AppConfig:
    load_env()
    root = find_project_root()
    paths = Paths(
        data_dir=_resolve_path(root, os.environ.get("DATA_DIR", "data")),
        raw_data_dir=_resolve_path(root, os.environ.get("RAW_DATA_DIR", "data/raw")),
        interim_data_dir=_resolve_path(root, os.environ.get("INTERIM_DATA_DIR", "data/interim")),
        processed_data_dir=_resolve_path(
            root, os.environ.get("PROCESSED_DATA_DIR", "data/processed")
        ),
        artifacts_dir=_resolve_path(root, os.environ.get("ARTIFACTS_DIR", "artifacts")),
        reports_dir=_resolve_path(root, os.environ.get("REPORTS_DIR", "reports")),
    )
    return AppConfig(
        project_root=root,
        paths=paths,
        random_seed=int(os.environ.get("RANDOM_SEED", "42")),
        default_config=_resolve_path(
            root, os.environ.get("DEFAULT_CONFIG", "configs/baseline.yaml")
        ),
        kaggle_competition=os.environ.get("KAGGLE_COMPETITION", "nlp-getting-started"),
        log_level=os.environ.get("LOG_LEVEL", "INFO"),
        app_env=os.environ.get("APP_ENV", "development"),
    )


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    return build_config()
