"""
Merge multimodal analyst CSV outputs into a unified incident master database.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

MAIN_DIR = Path(__file__).resolve().parents[1]
INTEGRATION_DIR = Path(__file__).resolve().parent

AUDIO_CSV = MAIN_DIR / "audio" / "911_audio_analysis.csv"
DOCUMENT_CSV = MAIN_DIR / "document" / "structured_incident_reports.csv"
IMAGE_CSV = MAIN_DIR / "images" / "image_analysis_output.csv"
TEXT_CSV = MAIN_DIR / "text" / "text_analyst_output.csv"

MASTER_CSV = INTEGRATION_DIR / "master_incidents.csv"

HIGH_EVENT_KEYWORDS = (
    "fire", "shooting", "gun", "homicide", "murder", "death", "stabbing",
    "assault", "trapped", "drowning", "explosion", "hostage", "robbery",
)
MEDIUM_EVENT_KEYWORDS = (
    "traffic", "collision", "accident", "theft", "disturbance", "rescue",
    "medical", "investigation", "smoke", "distress",
)
LOW_EVENT_KEYWORDS = (
    "training", "proposal", "application", "non-emergency", "unknown", "no-fire",
)

SOURCE_ID_PREFIX = {
    "Audio": "AUD",
    "PDF": "DOC",
    "Image": "IMG",
    "Text": "TXT",
}


def format_incident_id(source: str, index: int) -> str:
    """Build assignment-style IDs such as AUD-001, DOC-001, IMG-001, TXT-001."""
    prefix = SOURCE_ID_PREFIX[source]
    width = max(3, len(str(index)))
    return f"{prefix}-{index:0{width}d}"


def _clean(value: object, default: str = "Unknown") -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    text = str(value).strip()
    if not text or text.upper() in {"N/A", "NA", "NONE", "NAN"}:
        return default
    return text


def _event_severity_hint(event: str) -> str | None:
    lower = event.lower()
    if any(k in lower for k in HIGH_EVENT_KEYWORDS):
        return "High"
    if any(k in lower for k in MEDIUM_EVENT_KEYWORDS):
        return "Medium"
    if any(k in lower for k in LOW_EVENT_KEYWORDS):
        return "Low"
    return None


def _score_to_severity(score: float) -> str:
    if score >= 0.65:
        return "High"
    if score >= 0.35:
        return "Medium"
    return "Low"


def classify_audio_severity(row: pd.Series) -> str:
    urgency = pd.to_numeric(row.get("Urgency_Score"), errors="coerce")
    event_hint = _event_severity_hint(_clean(row.get("Extracted_Event"), ""))
    sentiment = _clean(row.get("Sentiment"), "").lower()

    if event_hint:
        if event_hint == "High":
            return "High"
        if event_hint == "Medium" and pd.notna(urgency) and urgency >= 0.5:
            return "High"

    if pd.notna(urgency):
        base = _score_to_severity(float(urgency))
        if base == "Medium" and sentiment in {"distressed", "panicked", "urgent"}:
            return "High"
        return base

    return event_hint or "Medium"


def classify_document_severity(row: pd.Series) -> str:
    incident_type = _clean(row.get("Incident_Type"), "")
    lower = incident_type.lower()
    if "training" in lower or "application" in lower or "proposal" in lower:
        return "Low"
    if "request" in lower:
        return "Medium"
    return _event_severity_hint(incident_type) or "Low"


def classify_image_severity(row: pd.Series) -> str:
    scene = _clean(row.get("Scene_Type"), "")
    objects = _clean(row.get("Objects_Detected"), "")
    confidence = pd.to_numeric(row.get("Confidence_Score"), errors="coerce")

    combined = f"{scene} {objects}".lower()
    if "fire" in combined:
        return "High"
    if "smoke" in combined:
        return "Medium"
    if pd.notna(confidence) and confidence <= 0:
        return "Low"
    return "Low"


def classify_text_severity(row: pd.Series) -> str:
    topic = _clean(row.get("Topic"), "")
    sentiment = _clean(row.get("Sentiment"), "").lower()
    topic_hint = _event_severity_hint(topic)

    if topic_hint == "High":
        return "High"
    if sentiment == "negative" and topic_hint == "Medium":
        return "High"
    if topic_hint:
        return topic_hint
    if sentiment == "negative":
        return "Medium"
    return "Low"


def normalize_audio(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for index, (_, row) in enumerate(df.iterrows(), start=1):
        call_id = _clean(row.get("Call_ID"), "UNKNOWN")
        rows.append(
            {
                "Incident_ID": format_incident_id("Audio", index),
                "Source": "Audio",
                "Event": _clean(row.get("Extracted_Event")),
                "Location": _clean(row.get("Location")),
                "Time": "Unknown",
                "Severity": classify_audio_severity(row),
                "Source_Record_ID": call_id,
                "Detail": _clean(row.get("Transcript"), "")[:280],
            }
        )
    return pd.DataFrame(rows)


def normalize_document(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for index, (_, row) in enumerate(df.iterrows(), start=1):
        report_id = _clean(row.get("Report_ID"), "UNKNOWN")
        date_raw = _clean(row.get("Date"), "Unknown")
        rows.append(
            {
                "Incident_ID": format_incident_id("PDF", index),
                "Source": "PDF",
                "Event": _clean(row.get("Incident_Type")),
                "Location": _clean(row.get("Location")),
                "Time": date_raw,
                "Severity": classify_document_severity(row),
                "Source_Record_ID": report_id,
                "Detail": _clean(row.get("Summary"), "")[:280],
            }
        )
    return pd.DataFrame(rows)


def normalize_image(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for index, (_, row) in enumerate(df.iterrows(), start=1):
        image_id = _clean(row.get("Image_ID"), "UNKNOWN")
        location = _clean(row.get("Text_Extracted"))
        if location == "Unknown":
            location = "Scene photograph"
        rows.append(
            {
                "Incident_ID": format_incident_id("Image", index),
                "Source": "Image",
                "Event": _clean(row.get("Scene_Type")),
                "Location": location,
                "Time": "Unknown",
                "Severity": classify_image_severity(row),
                "Source_Record_ID": image_id,
                "Detail": _clean(row.get("Objects_Detected"), "")[:280],
            }
        )
    return pd.DataFrame(rows)


def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for index, (_, row) in enumerate(df.iterrows(), start=1):
        text_id = _clean(row.get("Text_ID"), "UNKNOWN")
        location = _clean(row.get("Entities"))
        rows.append(
            {
                "Incident_ID": format_incident_id("Text", index),
                "Source": "Text",
                "Event": _clean(row.get("Topic")),
                "Location": location,
                "Time": "Unknown",
                "Severity": classify_text_severity(row),
                "Source_Record_ID": text_id,
                "Detail": _clean(row.get("Raw_Text"), "")[:280],
            }
        )
    return pd.DataFrame(rows)


def load_source_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label} CSV: {path}")
    return pd.read_csv(path)


def build_long_master(
    audio_df: pd.DataFrame,
    document_df: pd.DataFrame,
    image_df: pd.DataFrame,
    text_df: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize and stack all modality records into the assignment output schema."""
    parts = [
        normalize_audio(audio_df),
        normalize_document(document_df),
        normalize_image(image_df),
        normalize_text(text_df),
    ]
    return pd.concat(parts, ignore_index=True)


def export_master_tables() -> pd.DataFrame:
    audio_df = load_source_csv(AUDIO_CSV, "audio")
    document_df = load_source_csv(DOCUMENT_CSV, "document")
    image_df = load_source_csv(IMAGE_CSV, "image")
    text_df = load_source_csv(TEXT_CSV, "text")

    long_master = build_long_master(audio_df, document_df, image_df, text_df)

    display_cols = ["Incident_ID", "Source", "Event", "Location", "Time", "Severity"]
    long_master[display_cols + ["Source_Record_ID", "Detail"]].to_csv(MASTER_CSV, index=False)

    return long_master


def main() -> None:
    long_master = export_master_tables()
    print(f"Wrote {len(long_master)} unified records to {MASTER_CSV}")
    print("\nSeverity breakdown:")
    print(long_master["Severity"].value_counts().to_string())
    print("\nSource breakdown:")
    print(long_master["Source"].value_counts().to_string())


if __name__ == "__main__":
    main()
