from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path

import requests

from disaster_tweets.config import get_config

AFINN_URL = "https://raw.githubusercontent.com/fnielsen/afinn/master/afinn/data/AFINN-en-165.txt"
BING_POSITIVE_URL = (
    "https://raw.githubusercontent.com/jeffreybreen/twitter-sentiment-analysis-tutorial-201107/"
    "master/data/opinion-lexicon-English/positive-words.txt"
)
BING_NEGATIVE_URL = (
    "https://raw.githubusercontent.com/jeffreybreen/twitter-sentiment-analysis-tutorial-201107/"
    "master/data/opinion-lexicon-English/negative-words.txt"
)
REQUEST_TIMEOUT_SECONDS = 30


def _download_file(url: str, dest: Path) -> None:
    response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    dest.write_bytes(response.content)


def download_kaggle_competition(dest_dir: Path, competition: str, force: bool = False) -> None:
    train_path = dest_dir / "train.csv"
    if train_path.exists() and not force:
        print(
            f"train.csv already present at {train_path}, "
            f"skipping Kaggle download (use --force to redownload)"
        )
        return
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print(
            "The 'kaggle' package is not installed. Install project dependencies "
            "(pip install -e '.[dev]') and configure credentials, then retry."
        )
        return
    try:
        api = KaggleApi()
        api.authenticate()
    except Exception as exc:
        print(
            f"Kaggle authentication failed: {exc}. Configure KAGGLE_USERNAME/KAGGLE_KEY "
            f"or place a valid kaggle.json at ~/.kaggle/kaggle.json, then retry."
        )
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / f"{competition}.zip"
    try:
        api.competition_download_files(competition, path=str(dest_dir), quiet=False)
        if zip_path.exists():
            with zipfile.ZipFile(zip_path) as archive:
                archive.extractall(dest_dir)
            zip_path.unlink()
        print(f"Kaggle competition files downloaded to {dest_dir}")
    except Exception as exc:
        print(f"Kaggle download failed: {exc}")


def download_lexicons(dest_dir: Path, force: bool = False) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    targets = {
        "AFINN": (AFINN_URL, dest_dir / "afinn_en_165.txt"),
        "Bing positive": (BING_POSITIVE_URL, dest_dir / "bing_positive.txt"),
        "Bing negative": (BING_NEGATIVE_URL, dest_dir / "bing_negative.txt"),
    }
    for name, (url, path) in targets.items():
        if path.exists() and not force:
            print(f"{name} already present at {path}, skipping (use --force to redownload)")
            continue
        try:
            _download_file(url, path)
            print(f"{name} downloaded to {path}")
        except requests.RequestException as exc:
            print(f"{name} download failed: {exc}")

    nrc_path = dest_dir / "nrc_emotion_lexicon.txt"
    nrc_url = os.environ.get("NRC_LEXICON_URL")
    if nrc_path.exists() and not force:
        print(f"NRC already present at {nrc_path}, skipping (use --force to redownload)")
    elif nrc_url:
        try:
            _download_file(nrc_url, nrc_path)
            print(f"NRC lexicon downloaded to {nrc_path}")
        except requests.RequestException as exc:
            print(f"NRC download failed: {exc}")
    else:
        print(
            "NRC_LEXICON_URL is not set. The NRC Emotion Lexicon requires manual download "
            "from https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm after accepting "
            f"its academic-use license, then place the file at {nrc_path}."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the disaster tweets dataset and sentiment lexicons"
    )
    parser.add_argument("--dest", type=Path, default=None)
    parser.add_argument("--only", choices=["kaggle", "lexicons", "all"], default="all")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = get_config()
    raw_dir = args.dest or config.paths.raw_data_dir
    lexicons_dir = raw_dir / "lexicons"

    if args.only in ("kaggle", "all"):
        download_kaggle_competition(raw_dir, config.kaggle_competition, force=args.force)
    if args.only in ("lexicons", "all"):
        download_lexicons(lexicons_dir, force=args.force)


if __name__ == "__main__":
    main()
