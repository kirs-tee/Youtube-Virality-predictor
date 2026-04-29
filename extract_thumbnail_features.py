import requests
from PIL import Image
from io import BytesIO
import cv2
import numpy as np
import os
from tqdm import tqdm
import json
from fer.fer import FER
from mtcnn import MTCNN
from ultralytics import YOLO
import pytesseract
import easyocr

ocr_reader = easyocr.Reader(['en'], gpu=False)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"



INPUT_JSON = "youtube_gaming_dataset.json"
OUTPUT_JSON = "dataset_with_thumbnail_features.json"
THUMB_DIR = "thumbnails"
TEST_LIMIT = None  # FOR TESTING (set to None to process all)

emotion_detector = FER(mtcnn=True)


model = YOLO("yolov8n.pt")  # fast + good enough for thumbnails



#  Load and Save methods
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# Download the thumbnails method
def download_thumbnail(url, path):
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    img = Image.open(BytesIO(r.content)).convert("RGB")
    img.save(path, "JPEG")

def get_best_thumbnail_url(video_id):
    bases = [
        "maxresdefault.jpg",
        "hqdefault.jpg",
        "mqdefault.jpg",
        "default.jpg"
    ]
    for b in bases:
        url = f"https://i.ytimg.com/vi/{video_id}/{b}"
        r = requests.get(url, timeout=5)
        if r.status_code == 200 and len(r.content) > 5000:
            return url
    return None


# Thumbnail features 
def extract_thumbnail_features(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))

    edges = cv2.Canny(gray, 100, 200)
    edge_density = float(np.mean(edges > 0))

    return {
        "thumbnailBrightness": round(brightness, 2),
        "thumbnailContrast": round(contrast, 2),
        "thumbnailEdgeDensity": round(edge_density, 4)
    }

# Object detection
def analyse_thumbnail(image_path):
    img = cv2.imread(image_path)
    results = model(img, conf=0.7, verbose=False)[0]

    detected_classes = []
    person_count = 0

    for box in results.boxes:
        conf = float(box.conf.item())
        if conf < 0.7:
            continue

        label = model.names[int(box.cls.item())]
        detected_classes.append(label)

        if label == "person":
            person_count += 1

    unique_objects = sorted(set(detected_classes))

    return {
        "thumbnailHasPerson": person_count > 0,
        "thumbnailPersonCount": person_count,
        "thumbnailTopObjects": unique_objects[:5]
    }




# Extract emotions from the thumbnail
def extract_thumbnail_emotion(img_path):
    img = cv2.imread(img_path)

    if img is None:
        return {
            "thumbnailHasFace": 0,
            "thumbnailFaceCount": 0,
            "thumbnailDominantEmotion": None,
            "thumbnailEmotionConfidence": None,
            "thumbnailLargestFaceRatio": 0
        }

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = emotion_detector.detect_emotions(img_rgb)

    if not results:
        return {
            "thumbnailHasFace": 0,
            "thumbnailFaceCount": 0,
            "thumbnailDominantEmotion": None,
            "thumbnailEmotionConfidence": None,
            "thumbnailLargestFaceRatio": 0
        }

    image_area = img.shape[0] * img.shape[1]
    largest_face_area = 0

    best_emotion = None
    best_score = 0.0

    for face in results:
        # Emotion extraction
        for emotion, score in face["emotions"].items():
            if score > best_score:
                best_score = score
                best_emotion = emotion

        # Face area calculation
        x, y, w, h = face["box"]
        face_area = w * h
        if face_area > largest_face_area:
            largest_face_area = face_area

    largest_face_ratio = largest_face_area / image_area

    return {
        "thumbnailHasFace": 1,
        "thumbnailFaceCount": len(results),
        "thumbnailDominantEmotion": best_emotion,
        "thumbnailEmotionConfidence": round(best_score, 3),
        "thumbnailLargestFaceRatio": round(largest_face_ratio, 4)
    }


def extract_thumbnail_saturation(img_path):
    img = cv2.imread(img_path)

    if img is None:
        return 0

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Saturation channel is index 1
    saturation = hsv[:, :, 1]

    avg_saturation = saturation.mean() / 255  # normalize 0–1

    return round(avg_saturation, 4)



def extract_thumbnail_text_features(img_path):
    img = cv2.imread(img_path)

    if img is None:
        return {
            "thumbnailHasText": 0,
            "thumbnailWordCount": 0,
            "thumbnailTextRedRatio": 0
        }

    results = ocr_reader.readtext(img)

    words = []
    red_pixels = 0
    total_pixels = 0

    for (bbox, text, confidence) in results:
        if confidence > 0.7:
            words.extend(text.split())

            # Extract bounding box area
            pts = np.array(bbox).astype(int)
            x_min = np.min(pts[:, 0])
            x_max = np.max(pts[:, 0])
            y_min = np.min(pts[:, 1])
            y_max = np.max(pts[:, 1])

            text_region = img[y_min:y_max, x_min:x_max]

            if text_region.size > 0:
                # Calculate red dominance
                red = text_region[:, :, 2]
                green = text_region[:, :, 1]
                blue = text_region[:, :, 0]

                red_pixels += np.sum(red > (green + 20))  # red significantly higher
                total_pixels += red.size

    red_ratio = red_pixels / total_pixels if total_pixels > 0 else 0

    return {
        "thumbnailHasText": 1 if len(words) > 0 else 0,
        "thumbnailWordCount": len(words),
        "thumbnailTextRedRatio": round(red_ratio, 4)
    }


def detect_arrow(img_path):
    img = cv2.imread(img_path)

    if img is None:
        return 0

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Ignore tiny contours
        if area < 4000:
            continue

        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)

        # Arrows usually have 5–10 vertices due to tip + body
        if 5 <= len(approx) <= 12:
            
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)

            # Must be elongated
            if aspect_ratio > 1.3 or aspect_ratio < 0.75:
                return 1

    return 0


# processing dataset
def process_dataset():
    os.makedirs(THUMB_DIR, exist_ok=True)

    # Load fresh API dataset (latest metrics)
    raw_data = load_json(INPUT_JSON)

    # Load existing processed dataset
    if os.path.exists(OUTPUT_JSON):
        existing_data = load_json(OUTPUT_JSON)
        print(f"Existing dataset loaded: {len(existing_data)} entries")
    else:
        existing_data = []
        print("No existing dataset found. Starting fresh.")

    # Convert existing dataset to dictionary for fast lookup
    existing_dict = {row["videoId"]: row for row in existing_data}

    updated_output = []

    for row in tqdm(raw_data, desc="Updating dataset"):
        video_id = row["videoId"]

        if video_id in existing_dict:
            # UPDATE METRICS ONLY
            existing_row = existing_dict[video_id]

            # Update dynamic fields
            for key in row:
                existing_row[key] = row[key]

            updated_output.append(existing_row)

        else:
            # NEW VIDEO → FULL PROCESS
            print(f"Processing new video: {video_id}")

            url = get_best_thumbnail_url(video_id)
            thumb_path = os.path.join(THUMB_DIR, f"{video_id}.jpg")

            try:
                if url:
                    download_thumbnail(url, thumb_path)

                    visual_features = extract_thumbnail_features(thumb_path)
                    emotion_features = extract_thumbnail_emotion(thumb_path)
                    object_features = analyse_thumbnail(thumb_path)
                    saturation_value = extract_thumbnail_saturation(thumb_path)
                    text_features = extract_thumbnail_text_features(thumb_path)
                    arrow_flag = detect_arrow(thumb_path)

                    row.update(visual_features)
                    row.update(emotion_features)
                    row.update(object_features)
                    row["thumbnailAvgSaturation"] = saturation_value
                    row.update(text_features)
                    row["thumbnailHasArrow"] = arrow_flag
                    row["thumbnailFilename"] = f"{video_id}.jpg"

            except Exception as e:
                print(f"Failed for {video_id}: {e}")

            updated_output.append(row)

    print(f"\n Final dataset size: {len(updated_output)}")

    save_json(OUTPUT_JSON, updated_output)


if __name__ == "__main__":
    process_dataset()


