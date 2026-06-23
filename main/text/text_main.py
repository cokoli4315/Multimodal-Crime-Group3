#!/usr/bin/env python3
"""
Text Analyst – Multimodal Crime / Incident Report Analyzer
Tools: spaCy, HuggingFace Transformers, NLTK
Dataset: CrimeReport Twitter Dataset (Kaggle)
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import nltk
import pandas as pd
import spacy
from transformers import pipeline

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SCRIPT_DIR / "text_analyst_output.csv"
DEFAULT_RAW_CSV = SCRIPT_DIR / "crimereport.csv"

CANDIDATE_LABELS = [
    "Road Accident",
    "Fire",
    "Theft / Robbery",
    "Assault / Shooting",
    "Public Disturbance",
    "Missing Person",
    "Arrest / Investigation",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run NLP analysis on crime-related text and export structured CSV."
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Input file (.txt JSONL from Kaggle or .csv with a text column).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT.name}).",
    )
    parser.add_argument(
        "--download-kaggle",
        action="store_true",
        help="Download the CrimeReport dataset from Kaggle into this folder.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only the first N records (0 = all). Useful for quick tests.",
    )
    return parser.parse_args()


def ensure_nltk_data() -> None:
    for resource in ("stopwords", "punkt", "punkt_tab"):
        nltk.download(resource, quiet=True)


def ensure_spacy_model() -> None:
    try:
        spacy.load("en_core_web_sm")
    except OSError:
        print("Downloading spaCy model en_core_web_sm ...")
        subprocess.run(
            [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
            check=True,
        )


def download_kaggle_dataset(target_dir: Path) -> Path:
    try:
        subprocess.run(
            [
                "kaggle",
                "datasets",
                "download",
                "-d",
                "cameliasiadat/crimereport",
                "--unzip",
                "-p",
                str(target_dir),
            ],
            check=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Kaggle CLI not found. Install with: pip install kaggle\n"
            "Then place your API token at ~/.kaggle/kaggle.json"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Kaggle download failed. Ensure ~/.kaggle/kaggle.json is configured."
        ) from exc

    candidates = list(target_dir.glob("CrimeReport*.txt")) + list(target_dir.glob("*.txt"))
    if not candidates:
        raise FileNotFoundError(f"No CrimeReport .txt file found in {target_dir}")
    return candidates[0]


def resolve_input_path(args: argparse.Namespace) -> Path:
    if args.download_kaggle:
        return download_kaggle_dataset(SCRIPT_DIR)

    if args.input:
        path = args.input.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        return path

    for candidate in (
        SCRIPT_DIR / "CrimeReport (1).txt",
        SCRIPT_DIR / "CrimeReport.txt",
        DEFAULT_RAW_CSV,
    ):
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "No input file found. Provide --input, use --download-kaggle, or place one of:\n"
        "  - CrimeReport.txt / CrimeReport (1).txt\n"
        "  - crimereport.csv\n"
        f"in {SCRIPT_DIR}"
    )


def load_jsonl(path: Path) -> pd.DataFrame:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not records:
        raise ValueError(f"No valid JSON records found in {path}")

    df_raw = pd.DataFrame(records)
    df = pd.DataFrame()
    df["Raw_Text"] = df_raw["text"]
    df["Created_At"] = df_raw.get("created_at", "Unknown")
    df["Location"] = df_raw.get("place", pd.Series(["Unknown"] * len(df_raw))).apply(
        lambda x: x["full_name"] if isinstance(x, dict) else "Unknown"
    )
    df["Username"] = df_raw.get("user", pd.Series(["Unknown"] * len(df_raw))).apply(
        lambda x: x["screen_name"] if isinstance(x, dict) else "Unknown"
    )
    df["Source"] = "Twitter"
    return df


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    text_col = next(
        (col for col in ("Raw_Text", "text", "Text", "tweet") if col in df.columns),
        None,
    )
    if text_col is None:
        raise ValueError(
            f"CSV must contain a text column (Raw_Text, text, Text, or tweet). "
            f"Found: {list(df.columns)}"
        )

    df = df.copy()
    df["Raw_Text"] = df[text_col]
    df["Source"] = df["Source"] if "Source" in df.columns else "Twitter"
    return df


def load_input(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return load_csv(path)
    return load_jsonl(path)


def clean_text(text: object) -> str:
    text = str(text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_entities(text: object, nlp: spacy.Language) -> str:
    doc = nlp(str(text))
    relevant = [
        ent.text
        for ent in doc.ents
        if ent.label_ in {"GPE", "LOC", "PERSON", "ORG", "DATE"}
    ]
    return ", ".join(relevant) if relevant else "None"


def get_sentiment(text: object, sentiment_pipeline) -> str:
    try:
        result = sentiment_pipeline(str(text)[:512])[0]
        return result["label"]
    except Exception:
        return "UNKNOWN"


def classify_topic(text: object, classifier) -> str:
    try:
        result = classifier(str(text)[:512], CANDIDATE_LABELS)
        return result["labels"][0]
    except Exception:
        return "Unknown"


def run_pipeline(df: pd.DataFrame, limit: int = 0) -> pd.DataFrame:
    if limit > 0:
        df = df.head(limit).copy()

    df["Text_ID"] = [f"TXT_{i:03d}" for i in range(1, len(df) + 1)]
    df["Clean_Text"] = df["Raw_Text"].apply(clean_text)

    print("Loading spaCy model ...")
    nlp = spacy.load("en_core_web_sm")
    df["Entities"] = df["Clean_Text"].apply(lambda text: extract_entities(text, nlp))

    print("Running sentiment analysis (this may take a few minutes) ...")
    sentiment_pipeline = pipeline("sentiment-analysis")
    df["Sentiment"] = df["Clean_Text"].apply(lambda text: get_sentiment(text, sentiment_pipeline))

    print("Running topic classification (this may take a few minutes) ...")
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    df["Topic"] = df["Clean_Text"].apply(lambda text: classify_topic(text, classifier))

    final_df = df[["Text_ID", "Source", "Raw_Text", "Sentiment", "Entities", "Topic"]].copy()
    return final_df


def main() -> None:
    args = parse_args()
    ensure_nltk_data()
    ensure_spacy_model()

    input_path = resolve_input_path(args)
    print(f"Loading input: {input_path}")
    df = load_input(input_path)
    print(f"Loaded {len(df)} records")

    final_df = run_pipeline(df, limit=args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(args.output, index=False)

    print("=" * 50)
    print("TEXT ANALYST PIPELINE — FINAL SUMMARY")
    print("=" * 50)
    print(f"Total Tweets Processed : {len(final_df)}")
    print(f"Sentiment Distribution :\n{final_df['Sentiment'].value_counts()}")
    print(f"\nTopic Distribution     :\n{final_df['Topic'].value_counts()}")
    print(f"\nOutput File            : {args.output}")
    print("=" * 50)


if __name__ == "__main__":
    main()
