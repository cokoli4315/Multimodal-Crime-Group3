import os
import re
import argparse
from pathlib import Path

import cv2
import pandas as pd
import pytesseract
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient
from ultralytics import YOLO


load_dotenv()


def clean_ocr_text(text):
    text = text.replace("\n", " ")
    text = re.sub(r"[^A-Za-z0-9\s\-]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text if text else "N/A"


def extract_text_from_image(image_path):
    image = cv2.imread(str(image_path))

    if image is None:
        return "N/A"

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    try:
        text = pytesseract.image_to_string(gray, config="--psm 6")
        return clean_ocr_text(text)
    except Exception:
        return "N/A"


def run_fire_smoke_detection(image_path, roboflow_client, model_id):
    detections = []

    try:
        result = roboflow_client.infer(str(image_path), model_id=model_id)

        for pred in result.get("predictions", []):
            label = pred.get("class", "unknown").lower()
            confidence = float(pred.get("confidence", 0.0))

            detections.append({
                "label": label,
                "confidence": confidence
            })

    except Exception as e:
        print(f"Roboflow detection failed for {image_path.name}: {e}")

    return detections


def run_yolov8_coco_detection(image_path, coco_model):
    allowed_objects = {
        "person",
        "car",
        "truck",
        "bus",
        "motorcycle",
        "bicycle"
    }

    detections = []

    try:
        results = coco_model(str(image_path), conf=0.25, verbose=False)

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                label = coco_model.names[class_id].lower()
                confidence = float(box.conf[0])

                if label in allowed_objects:
                    detections.append({
                        "label": label,
                        "confidence": confidence
                    })

    except Exception as e:
        print(f"YOLOv8 COCO detection failed for {image_path.name}: {e}")

    return detections


def classify_scene(detections):
    labels = {item["label"] for item in detections}

    has_fire = "fire" in labels
    has_smoke = "smoke" in labels
    has_vehicle = bool(labels.intersection({"car", "truck", "bus", "motorcycle", "bicycle"}))
    has_person = "person" in labels

    if has_fire and has_smoke:
        return "Fire and Smoke Scene"
    elif has_fire:
        return "Fire Scene"
    elif has_smoke:
        return "Smoke Scene"
    elif has_vehicle and has_person:
        return "Accident / Traffic Scene"
    elif has_vehicle:
        return "Vehicle Scene"
    elif has_person:
        return "Person Scene"
    else:
        return "No-fire / Unknown Scene"


def format_objects(detections):
    if not detections:
        return "N/A"

    best_confidence = {}

    for item in detections:
        label = item["label"]
        confidence = item["confidence"]

        if label not in best_confidence or confidence > best_confidence[label]:
            best_confidence[label] = confidence

    return "; ".join(
        f"{label}:{confidence:.2f}"
        for label, confidence in sorted(best_confidence.items())
    )


def get_confidence_score(detections):
    if not detections:
        return 0.00

    return round(max(item["confidence"] for item in detections), 2)


def main():
    parser = argparse.ArgumentParser(description="Student 3 Image Analyst Pipeline")

    parser.add_argument(
        "--image_dir",
        default="sample_images",
        help="Folder containing input images"
    )

    parser.add_argument(
        "--output_csv",
        default="image_analysis_output.csv",
        help="Output CSV file path"
    )

    parser.add_argument(
        "--roboflow_model_id",
        default="fire-smoke-yolov8/1",
        help="Roboflow model ID copied from Roboflow API Docs"
    )

    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    output_csv = Path(args.output_csv)

    api_key = os.getenv("ROBOFLOW_API_KEY")

    if not api_key:
        raise ValueError(
            "ROBOFLOW_API_KEY is missing. Add it to a .env file or set it as an environment variable."
        )

    if not image_dir.exists():
        raise FileNotFoundError(f"Image folder not found: {image_dir}")

    roboflow_client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=api_key
    )

    coco_model = YOLO("yolov8n.pt")

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = [
        path for path in image_dir.iterdir()
        if path.suffix.lower() in image_extensions
    ]

    rows = []

    for idx, image_path in enumerate(sorted(image_paths), start=1):
        print(f"Processing {image_path.name}")

        fire_smoke_detections = run_fire_smoke_detection(
            image_path,
            roboflow_client,
            args.roboflow_model_id
        )

        coco_detections = run_yolov8_coco_detection(image_path, coco_model)

        all_detections = fire_smoke_detections + coco_detections

        row = {
            "Image_ID": f"IMG_{idx:03d}",
            "Scene_Type": classify_scene(all_detections),
            "Objects_Detected": format_objects(all_detections),
            "Text_Extracted": extract_text_from_image(image_path),
            "Confidence_Score": get_confidence_score(all_detections)
        }

        rows.append(row)

    df = pd.DataFrame(rows, columns=[
        "Image_ID",
        "Scene_Type",
        "Objects_Detected",
        "Text_Extracted",
        "Confidence_Score"
    ])

    df.to_csv(output_csv, index=False)
    print(f"Saved image analysis output to {output_csv}")


if __name__ == "__main__":
    main()