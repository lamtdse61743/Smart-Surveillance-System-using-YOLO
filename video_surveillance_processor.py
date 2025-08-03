import cv2
import time
import csv
import os
import uuid
import torch
import numpy as np
from datetime import datetime
from collections import deque
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
import mediapipe as mp

base_dir = os.path.dirname(os.path.abspath(__file__))
box_model_path = os.path.join(base_dir, "models", "box_yolov9t.pt")
general_model_path = os.path.join(base_dir, "models", "yolov9t.pt")
log_path = os.path.join(base_dir, "log", "log.csv")
progress_file = os.path.join(base_dir, "log", "progress.txt")

# Load YOLO models
try:
    model_box = YOLO(box_model_path)
    model_general = YOLO(general_model_path)
except Exception as e:
    print(f"Error loading YOLO models: {e}")
    model_box = None
    model_general = None

focal_px = 700
frame_skip = 5
person_proximity_cooldown_sec = 10

target_classes = {0: "person", 2: "car", 16: "cat", 17: "dog", 80: "box"}
real_height_m = {0: 1.7, 2: 1.4, 16: 0.1, 17: 0.1, 80: 0.1}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
face_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
mp_face = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.7)

def get_embedding(img):
    if img.size == 0 or img.shape[0] < 10 or img.shape[1] < 10:
        print("Warning: Invalid face crop for embedding")
        return np.zeros(512)
    try:
        face = cv2.resize(img, (160, 160))[:, :, ::-1].copy()
        face = torch.tensor(face.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(device)
        face = (face - 127.5) / 128.0
        with torch.no_grad():
            emb = face_model(face)
        return emb[0].cpu().numpy() / np.linalg.norm(emb[0].cpu().numpy())
    except Exception as e:
        print(f"Error in get_embedding: {e}")
        return np.zeros(512)

def load_known_face(name, filepath):
    img = cv2.imread(filepath)
    if img is None:
        print(f"Error: Cannot load image {filepath}")
        return name, np.zeros(512)
    result = mp_face.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if result.detections:
        bbox = result.detections[0].location_data.relative_bounding_box
        h, w = img.shape[:2]
        x, y = int(bbox.xmin * w), int(bbox.ymin * h)
        w_box = int(bbox.width * w)
        h_box = int(bbox.height * h)
        face_crop = img[y:y+h_box, x:x+w_box]
        return name, get_embedding(face_crop)
    print(f"Warning: No face detected in {filepath}")
    return name, np.zeros(512)

known_faces = dict([
    load_known_face("Lam", os.path.join(base_dir, "home_owner_imgs", "Lam", "Lam.jpg")),
    load_known_face("William", os.path.join(base_dir, "home_owner_imgs", "William", "William.jpeg"))
])

os.makedirs("log", exist_ok=True)

def process_video(input_path, output_path):
    print(f"Processing video: {input_path} -> {output_path}")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {input_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width, height = int(cap.get(3)), int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    log_buffer = []
    recent_persons = deque()
    last_person_proximity_time = [0]  # Use list for consistency
    home_arrivals = set()
    persistent_boxes = []
    persistent_ttl = 5

    def notify_local(title, message):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_buffer.append([
            str(uuid.uuid4()), "N/A", "Alert", title, message, "", now, ""
        ])

    # Write log header if file doesn't exist
    if not os.path.exists(log_path):
        with open(log_path, mode="w", newline="") as log_file:
            log_writer = csv.writer(log_file)
            log_writer.writerow(["Event ID", "Frame", "Behavior", "Class", "Distance (m)", "Timestamp (s)", "Event Time (system)", "Closest Person Distance (m)"])

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        progress = int((frame_num / total_frames) * 100)
        with open(progress_file, "w") as f:
            f.write(str(progress))

        timestamp = frame_num / fps
        annotated = frame.copy()
        current_time = time.time()

        # Face Detection
        result_face = mp_face.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        best_match = "Unknown"
        best_score = 0.0
        if result_face.detections:
            for detection in result_face.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w = frame.shape[:2]
                x, y = int(bbox.xmin * w), int(bbox.ymin * h)
                w_box = int(bbox.width * w)
                h_box = int(bbox.height * h)
                face_crop = frame[y:y+h_box, x:x+w_box]
                if face_crop.size == 0:
                    continue
                try:
                    emb = get_embedding(face_crop)
                    for name, known_emb in known_faces.items():
                        score = np.dot(emb, known_emb)
                        if score > best_score:
                            best_match, best_score = name, score
                except Exception as e:
                    print(f"Face detection error: {e}")

                label = best_match if best_score > 0.7 else "Unknown"
                cv2.rectangle(annotated, (x, y), (x + w_box, y + h_box), (0, 0, 255), 2)
                cv2.putText(annotated, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                if label != "Unknown" and label not in home_arrivals:
                    home_arrivals.add(label)
                    log_buffer.append([
                        str(uuid.uuid4()), frame_num, "Door Open", label, "", round(timestamp, 2), datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ""
                    ])
                    notify_local("Home Owner Detected", f"{label} just came home!")

        persistent_boxes = [(b, l, t - 1) for (b, l, t) in persistent_boxes if t > 1]

        if frame_num % frame_skip == 0:
            if model_general:
                detections = model_general.track(frame, persist=True, verbose=False, tracker="bytetrack.yaml")[0]
                if detections.boxes.id is not None:
                    for box, cls_id, track_id in zip(detections.boxes.xyxy, detections.boxes.cls, detections.boxes.id):
                        cls_id = int(cls_id)
                        if cls_id not in target_classes:
                            continue

                        x1, y1, x2, y2 = map(int, box.tolist())
                        label = target_classes[cls_id]
                        box_height = y2 - y1
                        height_m = real_height_m.get(cls_id, 1.0)
                        distance_m = (focal_px * height_m) / box_height if box_height > 0 else None
                        text = f"{label}: {distance_m:.2f} m" if distance_m else f"{label}"

                        persistent_boxes.append(((x1, y1, x2, y2), text, persistent_ttl))

                        if cls_id == 0 and distance_m and distance_m < 5.0:
                            if current_time - last_person_proximity_time[0] > person_proximity_cooldown_sec:
                                notify_local("Proximity Alert", f"Person detected at {distance_m:.2f} meters")
                                log_buffer.append([
                                    str(uuid.uuid4()), frame_num, "Proximity Alert", label, round(distance_m, 2), round(timestamp, 2), datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ""
                                ])
                                recent_persons.append((track_id, distance_m, frame_num))
                                last_person_proximity_time[0] = current_time

        for (x1, y1, x2, y2), text, _ in persistent_boxes:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Fixed RMSE2 to 2
            cv2.putText(annotated, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        out.write(annotated)

    cap.release()
    out.release()

    # Append logs instead of overwriting
    with open(log_path, mode="a", newline="") as log_file:
        log_writer = csv.writer(log_file)
        if os.path.getsize(log_path) == 0:  # Write header only if file is empty
            log_writer.writerow(["Event ID", "Frame", "Behavior", "Class", "Distance (m)", "Timestamp (s)", "Event Time (system)", "Closest Person Distance (m)"])
        for entry in log_buffer:
            log_writer.writerow(entry)

    return os.path.basename(output_path)

def process_frame(frame, frame_num, fps, log_buffer, recent_persons, last_person_proximity_time, home_arrivals, persistent_boxes, emit_callback=None):
    timestamp = frame_num / fps
    current_time = time.time()
    annotated = frame.copy()

    def notify_local(title, message):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = [str(uuid.uuid4()), frame_num, "Alert", title, message, "", now, ""]
        log_buffer.append(entry)
        if emit_callback:
            emit_callback(entry)

    # Face Detection
    result_face = mp_face.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    best_match = "Unknown"
    best_score = 0.0
    if result_face.detections:
        for detection in result_face.detections:
            bbox = detection.location_data.relative_bounding_box
            h, w = frame.shape[:2]
            x, y = int(bbox.xmin * w), int(bbox.ymin * h)
            w_box = int(bbox.width * w)
            h_box = int(bbox.height * h)
            face_crop = frame[y:y+h_box, x:x+w_box]
            if face_crop.size == 0:
                continue
            try:
                emb = get_embedding(face_crop)
                for name, known_emb in known_faces.items():
                    score = np.dot(emb, known_emb)
                    if score > best_score:
                        best_match, best_score = name, score
            except Exception as e:
                print(f"Face detection error: {e}")

            label = best_match if best_score > 0.7 else "Unknown"
            cv2.rectangle(annotated, (x, y), (x + w_box, y + h_box), (0, 0, 255), 2)
            cv2.putText(annotated, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            if label != "Unknown" and label not in home_arrivals:
                home_arrivals.add(label)
                log_buffer.append([
                    str(uuid.uuid4()), frame_num, "Door Open", label, "", round(timestamp, 2), datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ""
                ])
                notify_local("Home Owner Detected", f"{label} just came home!")

    persistent_boxes[:] = [(b, l, t - 1) for (b, l, t) in persistent_boxes if t > 1]

    if frame_num % frame_skip == 0:
        if model_general:
            detections = model_general.track(frame, persist=True, verbose=False, tracker="bytetrack.yaml")[0]
            if detections.boxes.id is not None:
                for box, cls_id, track_id in zip(detections.boxes.xyxy, detections.boxes.cls, detections.boxes.id):
                    cls_id = int(cls_id)
                    if cls_id not in target_classes:
                        continue

                    x1, y1, x2, y2 = map(int, box.tolist())
                    label = target_classes[cls_id]
                    box_height = y2 - y1
                    height_m = real_height_m.get(cls_id, 1.0)
                    distance_m = (focal_px * height_m) / box_height if box_height > 0 else None
                    text = f"{label}: {distance_m:.2f} m" if distance_m else f"{label}"

                    persistent_boxes.append(((x1, y1, x2, y2), text, 5))

                    if cls_id == 0 and distance_m and distance_m < 5.0:
                        if current_time - last_person_proximity_time[0] > person_proximity_cooldown_sec:
                            notify_local("Proximity Alert", f"Person detected at {distance_m:.2f} meters")
                            log_buffer.append([
                                str(uuid.uuid4()), frame_num, "Proximity Alert", label, round(distance_m, 2), round(timestamp, 2), datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ""
                            ])
                            recent_persons.append((track_id, distance_m, frame_num))
                            last_person_proximity_time[0] = current_time

    for (x1, y1, x2, y2), text, _ in persistent_boxes:
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Fixed RMSE2 to 2
        cv2.putText(annotated, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return annotated, log_buffer, recent_persons, last_person_proximity_time, home_arrivals, persistent_boxes