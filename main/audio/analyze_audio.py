#!/usr/bin/env python3
"""Transcribe 911 WAV files and export structured emergency-call analysis."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import librosa
import pandas as pd
import torch
import whisper


EVENT_KEYWORDS = {
    "Building fire": (
        "building fire", "house fire", "apartment fire", "fatal fire", "fire rescue",
        "on fire", "fire", "smoke", "flames", "burning", "carbon monoxide",
    ),
    "Shooting / gun violence": (
        "shooting", "shot", "shots fired", "gunshot", "gunfire", "gun", "firearm",
        "shooter", "rifle", "shotgun", "armed",
    ),
    "Stabbing / assault": (
        "stabbing", "stabbed", "knife", "assault", "attacked", "attack", "beating",
        "fight", "choking", "strangled", "bitten",
    ),
    "Homicide / death investigation": (
        "murder", "murders", "murdered", "killed", "homicide", "dead body", "death", "fatality",
        "body found", "confession",
    ),
    "Medical emergency": (
        "unconscious", "not breathing", "can't breathe", "cannot breathe", "heart attack",
        "seizure", "overdose", "bleeding", "injured", "ambulance", "medical", "hospital",
        "diabetes", "choking", "collapsed", "cpr", "baby delivery", "childbirth",
        "post surgery", "fell ill", "became ill", "breathing problem", "stop breathing",
        "stopped breathing", "man down", "baby revived", "pregnant mother",
    ),
    "Traffic collision": (
        "car accident", "train accident", "interstate accident", "wrong-way accident",
        "plane crash", "crash", "crashed", "collision", "collisions", "pile-up",
        "hit by", "ran over", "run over", "vehicle", "truck", "car into",
        "out-of-control car", "dui",
    ),
    "Water emergency / drowning": (
        "drowning", "drowned", "fell into a pool", "pool", "boat rescue", "water rescue",
        "car into pond", "car in water",
    ),
    "Burglary / robbery / intruder": (
        "burglary", "break in", "broke in", "intruder", "robbery", "stealing", "stolen",
        "home invasion", "break into", "breaking into", "burglarize", "burglars", "carjacking",
        "purse snatch",
    ),
    "Domestic disturbance": (
        "domestic", "husband", "wife", "boyfriend", "girlfriend", "disturbance",
        "family dispute", "neighbor dispute",
    ),
    "Missing person / abduction": (
        "missing", "kidnap", "kidnapped", "kidnapping", "abducted", "abduction",
        "lost child", "missing baby", "can't find",
    ),
    "Suicide / self-harm": ("suicide", "suicidal", "self-harm", "kill myself", "jumping"),
    "Rescue / person trapped": (
        "trapped", "rescue", "collapse", "collapsed building", "stuck", "home alone",
    ),
    "Animal attack": ("dog attack", "bear attack", "animal attack", "bitten by", "bee swarm"),
    "Weather / environmental emergency": (
        "tornado", "flood", "flooded", "lightning strike", "snowstorm", "storm",
        "carbon monoxide", "gas leak",
    ),
    "Fall / accidental injury": (
        "fell from", "fall from", "manhole fall", "fell into", "crushed", "burn victim",
        "burned herself", "collapse",
    ),
    "Sexual assault / stalking": (
        "rape", "sexual assault", "stalking", "stalked", "peephole",
    ),
    "Child welfare / unattended child": (
        "child in car", "baby left", "home alone", "casino child", "left her child",
        "child was alone",
    ),
    "Threat / suspicious activity": (
        "threat", "threatening", "suspicious", "following me", "hostage", "help me",
        "man with gun", "welfare check",
    ),
    "Non-emergency / misuse of 911": (
        "non-emergency", "non-emerg", "prank", "false alarm", "call for a date",
        "nanny 911", "911 misuse", "bad sandwich", "burger king", "mcnugget",
        "fried rice", "bad chicken", "food order", "locked inside car", "sex 911",
        "mischief call",
    ),
    "Public safety / unusual incident": (
        "pursuit", "traffic stop", "air force one", "fly-over", "loose elephants",
        "rocket launcher", "school incident", "arrest incident",
        "wrong address", "delayed response", "dispatcher fired", "county commissioner",
        "mayor calls 911",
    ),
}

EVENT_SEVERITY = {
    "Shooting / gun violence": 0.88,
    "Stabbing / assault": 0.82,
    "Homicide / death investigation": 0.92,
    "Building fire": 0.82,
    "Medical emergency": 0.74,
    "Water emergency / drowning": 0.82,
    "Traffic collision": 0.68,
    "Missing person / abduction": 0.78,
    "Suicide / self-harm": 0.90,
    "Rescue / person trapped": 0.76,
    "Animal attack": 0.68,
    "Burglary / robbery / intruder": 0.65,
    "Domestic disturbance": 0.58,
    "Threat / suspicious activity": 0.62,
    "Non-emergency / misuse of 911": 0.08,
    "Weather / environmental emergency": 0.68,
    "Fall / accidental injury": 0.66,
    "Sexual assault / stalking": 0.82,
    "Child welfare / unattended child": 0.58,
    "Public safety / unusual incident": 0.38,
    "Other emergency / police assistance": 0.42,
}

STATE_ABBREVIATIONS = {
    "Ala": "Alabama", "Alaska": "Alaska", "Ariz": "Arizona", "Ark": "Arkansas",
    "Calif": "California", "Colo": "Colorado", "Conn": "Connecticut", "Del": "Delaware",
    "Fla": "Florida", "Ga": "Georgia", "Hawaii": "Hawaii", "Idaho": "Idaho",
    "Ill": "Illinois", "Ind": "Indiana", "Iowa": "Iowa", "Kan": "Kansas",
    "Ky": "Kentucky", "La": "Louisiana", "Maine": "Maine", "Md": "Maryland",
    "Mass": "Massachusetts", "Mich": "Michigan", "Minn": "Minnesota", "Miss": "Mississippi",
    "Mo": "Missouri", "Mont": "Montana", "Neb": "Nebraska", "Nev": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "Ohio": "Ohio", "Okla": "Oklahoma",
    "Ore": "Oregon", "Penn": "Pennsylvania", "Pa": "Pennsylvania", "RI": "Rhode Island",
    "SC": "South Carolina", "SD": "South Dakota", "Tenn": "Tennessee", "Tex": "Texas",
    "Utah": "Utah", "Vt": "Vermont", "Va": "Virginia", "Wash": "Washington",
    "WVa": "West Virginia", "Wisc": "Wisconsin", "Wyo": "Wyoming", "DC": "DC",
}

HIGH_URGENCY = (
    "not breathing", "can't breathe", "cannot breathe", "unconscious", "people trapped",
    "person trapped", "trapped", "bleeding", "shot", "shooting", "stabbed", "gun", "fire",
    "dying", "dead", "overdose", "help me", "hurry", "right now",
)
MEDIUM_URGENCY = (
    "emergency", "ambulance", "police", "injured", "accident", "crash", "smoke",
    "intruder", "break in", "threat", "scared", "afraid", "please",
)
DISTRESS_WORDS = (
    "help", "please", "hurry", "scared", "afraid", "terrified", "crying", "screaming",
    "oh my god", "can't", "cannot", "emergency", "dying", "dead", "shot", "fire",
)
CALM_WORDS = ("calm", "okay", "safe", "fine", "under control", "no emergency")

STREET_SUFFIX = r"(?:street|st\.?|avenue|ave\.?|road|rd\.?|boulevard|blvd\.?|drive|dr\.?|lane|ln\.?|highway|hwy\.?|court|ct\.?|parkway|pkwy\.?)"
LOCATION_PATTERNS = (
    re.compile(rf"\b(?:at|on|near|by|outside|inside)\s+([A-Z0-9][\w.'-]*(?:\s+[A-Z0-9][\w.'-]*){{0,4}}\s+{STREET_SUFFIX})\b", re.I),
    re.compile(r"\b(?:at|on|near|by|outside|inside)\s+(\d{1,6}\s+[A-Za-z0-9][\w.'-]*(?:\s+[A-Za-z0-9][\w.'-]*){0,5})\b", re.I),
    re.compile(r"\b(?:in|near)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b"),
)
METADATA_PLACE_PATTERN = re.compile(
    r"\b(?:in|near|outside|at)\s+"
    r"([A-Z][A-Za-z.'-]+(?:\s+(?:[A-Z][A-Za-z.'-]+|County)){0,3})\s*"
    r"\(([^)]+)\)"
)
METADATA_COUNTY_PATTERN = re.compile(r"\b([A-Z][A-Za-z.'-]+(?:\s+[A-Z][A-Za-z.'-]+)?\s+County)\b")
NAME_PATTERN = re.compile(
    r"\b(?:my name is|this is|i am|i'm|his name is|her name is)\s+"
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b",
    re.I,
)


def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    return text if text else "[No intelligible speech detected]"


def phrase_present(text: str, phrase: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text, flags=re.I) is not None


def score_events(text: str) -> dict[str, int]:
    return {
        event: sum(3 if " " in keyword else 1 for keyword in keywords if phrase_present(text, keyword))
        for event, keywords in EVENT_KEYWORDS.items()
    }


def extract_event(text: str) -> str:
    scores = score_events(text)
    event, score = max(scores.items(), key=lambda item: item[1])
    return event if score else "Unknown"


def infer_event(transcript: str, metadata_row: pd.Series) -> tuple[str, str, float]:
    transcript_scores = score_events(transcript)
    transcript_event, transcript_score = max(transcript_scores.items(), key=lambda item: item[1])
    metadata_text = f"{metadata_row.get('title', '')} {metadata_row.get('description', '')}"
    metadata_scores = score_events(metadata_text)
    metadata_event, metadata_score = max(metadata_scores.items(), key=lambda item: item[1])

    # Highly diagnostic spoken evidence wins. Otherwise the title/description supplies
    # the context that is commonly absent from a six-second clip.
    if transcript_score >= 3:
        return transcript_event, "Transcript", min(0.98, 0.70 + transcript_score * 0.04)
    if metadata_score > 0:
        source = "Transcript + metadata" if transcript_score and transcript_event == metadata_event else "Metadata"
        confidence = min(0.96, 0.68 + metadata_score * 0.035)
        return metadata_event, source, confidence
    if transcript_score > 0:
        return transcript_event, "Transcript", 0.62
    return "Other emergency / police assistance", "Fallback category", 0.35


def extract_location(text: str) -> str:
    if text.startswith("[No intelligible"):
        return "Unknown"
    for pattern in LOCATION_PATTERNS:
        match = pattern.search(text)
        if match:
            location = match.group(1).strip(" ,.")
            location = re.split(r"\b(?:and|but|because|there|where)\b", location, maxsplit=1, flags=re.I)[0]
            return location.strip(" ,.") or "Unknown"
    return "Unknown"


def normalize_state(value: object) -> str:
    if pd.isna(value) or not str(value).strip():
        return ""
    return str(value).strip()


def expand_state_abbreviation(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z]", "", value)
    return STATE_ABBREVIATIONS.get(cleaned, value.strip(" ."))


def infer_location(transcript: str, metadata_row: pd.Series) -> tuple[str, str, float]:
    spoken = extract_location(transcript)
    state = normalize_state(metadata_row.get("state"))
    if spoken != "Unknown":
        location = f"{spoken}, {state}" if state and state.lower() not in spoken.lower() else spoken
        return location, "Transcript" + (" + metadata state" if state else ""), 0.90

    description = str(metadata_row.get("description", "") or "")
    title = str(metadata_row.get("title", "") or "")
    place_match = METADATA_PLACE_PATTERN.search(description)
    if place_match:
        place = place_match.group(1).strip()
        inferred_state = state or expand_state_abbreviation(place_match.group(2))
        location = f"{place}, {inferred_state}" if inferred_state else place
        return location, "Metadata description", 0.82
    county_match = METADATA_COUNTY_PATTERN.search(description)
    if county_match:
        county = county_match.group(1)
        location = f"{county}, {state}" if state else county
        return location, "Metadata description", 0.76

    # Titles commonly encode a state abbreviation even when the state column is blank.
    for abbreviation, full_state in STATE_ABBREVIATIONS.items():
        if re.search(rf"(?<!\w){re.escape(abbreviation)}\.?(?!\w)", title, re.I):
            state = state or full_state
            break
    if state:
        return state, "Metadata state", 0.72
    return "Location not provided in source data", "Unavailable", 0.0


def extract_names(text: str) -> str:
    names = []
    for match in NAME_PATTERN.finditer(text):
        name = match.group(1).strip()
        if name.lower() not in {"the police", "the operator", "an emergency"}:
            names.append(name)
    return "; ".join(dict.fromkeys(names)) if names else "Unknown"


def urgency_phrases(text: str) -> list[str]:
    phrases = [phrase for phrase in (*HIGH_URGENCY, *MEDIUM_URGENCY) if phrase_present(text, phrase)]
    return list(dict.fromkeys(phrases))


def analyze_tone(
    text: str,
    metadata_row: pd.Series | None = None,
    inferred_event: str | None = None,
) -> tuple[str, float, str, str]:
    lowered = text.lower()
    high_hits = sum(phrase_present(lowered, phrase) for phrase in HIGH_URGENCY)
    medium_hits = sum(phrase_present(lowered, phrase) for phrase in MEDIUM_URGENCY)
    distress_hits = sum(phrase_present(lowered, word) for word in DISTRESS_WORDS)
    calm_hits = sum(phrase_present(lowered, word) for word in CALM_WORDS)
    exclamations = min(text.count("!"), 3)
    uppercase_tokens = sum(token.isupper() and len(token) > 2 for token in text.split())

    score = 0.10 + high_hits * 0.18 + medium_hits * 0.08
    score += min(distress_hits, 5) * 0.06 + exclamations * 0.03 + min(uppercase_tokens, 3) * 0.02

    source = "Transcript"
    # Metadata and inferred event are supporting signals because clips last only six seconds.
    if metadata_row is not None:
        if float(metadata_row.get("deaths_binary", 0) or 0) > 0:
            score += 0.08
        if float(metadata_row.get("potential_death", 0) or 0) > 0:
            score += 0.05
        if float(metadata_row.get("false_alarm", 0) or 0) > 0:
            score -= 0.05
        if inferred_event:
            event_floor = EVENT_SEVERITY.get(inferred_event, 0.42)
            score = max(score, event_floor * 0.72)
            source = "Transcript + incident context"

    if text.startswith("[No intelligible"):
        score = EVENT_SEVERITY.get(inferred_event or "", 0.42) * 0.72
        source = "Incident context; no intelligible speech"
    score = round(max(0.0, min(score, 1.0)), 2)

    if score >= 0.65 or distress_hits >= 3:
        sentiment = "Distressed"
    elif score >= 0.35 or distress_hits >= 1:
        sentiment = "Concerned"
    else:
        sentiment = "Calm"
    return sentiment, score, "; ".join(urgency_phrases(text)) or "None detected", source


def transcribe_wav(model, path: Path, language: str = "en") -> str:
    audio, _ = librosa.load(path, sr=whisper.audio.SAMPLE_RATE, mono=True)
    result = model.transcribe(
        audio,
        language=language,
        task="transcribe",
        fp16=False,
        temperature=0,
        condition_on_previous_text=False,
        verbose=False,
    )
    return normalize_text(result.get("text", ""))


def load_checkpoint(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def save_checkpoint(path: Path, transcripts: dict[str, str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(transcripts, handle, ensure_ascii=False, indent=2)
    temporary.replace(path)


def run_pipeline(
    dataset_dir: Path,
    output_csv: Path,
    detailed_csv: Path,
    checkpoint: Path,
    model_name: str,
    model_dir: Path = Path(".models/whisper"),
    limit: int | None = None,
) -> pd.DataFrame:
    metadata_path = dataset_dir / "911_metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    metadata = pd.read_csv(metadata_path)
    required = {"id", "filename"}
    if missing := required.difference(metadata.columns):
        raise ValueError(f"Metadata is missing required columns: {sorted(missing)}")
    if limit:
        metadata = metadata.head(limit)

    transcripts = load_checkpoint(checkpoint)
    unresolved = []
    for filename in metadata["filename"]:
        path = Path(filename)
        if not path.is_absolute() and not path.exists():
            path = dataset_dir.parent / path
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")
        key = str(path)
        if key not in transcripts:
            unresolved.append(path)

    model = None
    if unresolved:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Whisper {model_name!r} on {device}; {len(set(unresolved))} unique files remain.")
        model_dir.mkdir(parents=True, exist_ok=True)
        model = whisper.load_model(model_name, device=device, download_root=str(model_dir))
        for index, path in enumerate(dict.fromkeys(unresolved), start=1):
            try:
                transcripts[str(path)] = transcribe_wav(model, path)
            except Exception as exc:  # Keep the batch restartable if a single recording is damaged.
                transcripts[str(path)] = f"[Transcription error: {type(exc).__name__}]"
            if index % 10 == 0 or index == len(set(unresolved)):
                save_checkpoint(checkpoint, transcripts)
                print(f"Transcribed {index}/{len(set(unresolved))} remaining files.")

    rows = []
    detailed_rows = []
    for position, (_, row) in enumerate(metadata.iterrows(), start=1):
        path = Path(row["filename"])
        if not path.is_absolute() and not path.exists():
            path = dataset_dir.parent / path
        transcript = transcripts[str(path)]
        event, event_source, event_confidence = infer_event(transcript, row)
        location, location_source, location_confidence = infer_location(transcript, row)
        names = extract_names(transcript)
        sentiment, urgency, phrases, tone_source = analyze_tone(transcript, row, event)
        call_id = f"C{int(row['id']) + 1:03d}" if pd.notna(row["id"]) else f"C{position:03d}"

        output_row = {
            "Call_ID": call_id,
            "Transcript": transcript,
            "Extracted_Event": event,
            "Location": location,
            "Sentiment": sentiment,
            "Urgency_Score": urgency,
        }
        rows.append(output_row)
        detailed_rows.append({
            **output_row,
            "Extracted_Names": names,
            "Urgency_Phrases": phrases,
            "Audio_File": str(path),
            "Event_Source": event_source,
            "Event_Confidence": round(event_confidence, 2),
            "Location_Source": location_source,
            "Location_Confidence": round(location_confidence, 2),
            "Sentiment_Urgency_Source": tone_source,
            "Metadata_Title": str(row.get("title", "") or ""),
            "Metadata_Description": str(row.get("description", "") or ""),
        })

    result = pd.DataFrame(rows)
    result.to_csv(output_csv, index=False)
    pd.DataFrame(detailed_rows).to_csv(detailed_csv, index=False)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("911_first6sec"))
    parser.add_argument("--output", type=Path, default=Path("911_audio_analysis.csv"))
    parser.add_argument("--detailed-output", type=Path, default=Path("911_audio_analysis_detailed.csv"))
    parser.add_argument("--checkpoint", type=Path, default=Path("transcription_checkpoint.json"))
    parser.add_argument("--model", default="tiny.en", help="Whisper model name (default: tiny.en)")
    parser.add_argument("--model-dir", type=Path, default=Path(".models/whisper"))
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N metadata rows")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    frame = run_pipeline(
        dataset_dir=args.dataset_dir,
        output_csv=args.output,
        detailed_csv=args.detailed_output,
        checkpoint=args.checkpoint,
        model_name=args.model,
        model_dir=args.model_dir,
        limit=args.limit,
    )
    print(f"Wrote {len(frame)} rows to {args.output}")
    print(frame.head().to_string(index=False))
