def process_video(input_path, output_path):
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

    model_box = YOLO(box_model_path)
    model_general = YOLO(general_model_path)

    focal_px = 700
    frame_skip = 5
    person_proximity_cooldown_sec = 10

    target_classes = {0: "person", 2: "car", 16: "cat", 17: "dog", 80: "box"}
    real_height_m = {0: 1.7, 2: 1.4, 16: 0.1, 17: 0.1, 80: 0.1}

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    face_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    mp_face = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.7)

    def get_embedding(img):
        face = cv2.resize(img, (160, 160))[:, :, ::-1].copy()
        face = torch.tensor(face.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(device)
        face = (face - 127.5) / 128.0
        with torch.no_grad():
            emb = face_model(face)
        return emb[0].cpu().numpy() / np.linalg.norm(emb[0].cpu().numpy())

    def load_known_face(name, filepath):
        img = cv2.imread(filepath)
        if img is None:
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
        else:
            return name, np.zeros(512)

    known_faces = dict([
        load_known_face("Lam", os.path.join(base_dir, "home_owner_imgs", "Lam", "Lam.jpg")),
        load_known_face("William", os.path.join(base_dir, "home_owner_imgs", "William", "William.jpeg"))
    ])

    os.makedirs("log", exist_ok=True)
    csv_path = os.path.join(base_dir, "log", "face_processing_log.csv")
    alert_log_path = os.path.join(base_dir, "log", "alert_log.csv")
    progress_file = os.path.join(base_dir, "log", "progress.txt")

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = 30 if fps == 0 else fps
    width, height = int(cap.get(3)), int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    csv_log_buffer = []
    alert_log_buffer = []
    recent_persons = deque()
    last_person_proximity_time = 0
    home_arrivals = set()

    def notify_local(title, message):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        alert_log_buffer.append([now, title, message])

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
                    print("Face detection error:", e)

                label = best_match if best_score > 0.7 else "Unknown"
                cv2.rectangle(annotated, (x, y), (x + w_box, y + h_box), (0, 0, 255), 2)
                cv2.putText(annotated, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                if label != "Unknown" and label not in home_arrivals:
                    home_arrivals.add(label)
                    csv_log_buffer.append([str(uuid.uuid4()), frame_num, "Door Open", label, "", round(timestamp, 2), datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ""])
                    notify_local("Home Owner Detected", f"{label} just came home!")

        # Object Detection
        if frame_num % frame_skip == 0:
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
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    if cls_id == 0 and distance_m and distance_m < 5.0:
                        if current_time - last_person_proximity_time > person_proximity_cooldown_sec:
                            notify_local("Proximity Alert", f"Person detected at {distance_m:.2f} meters")
                            csv_log_buffer.append([str(uuid.uuid4()), frame_num, "Proximity Alert", label, round(distance_m, 2), round(timestamp, 2), datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ""])
                            recent_persons.append((track_id, distance_m, frame_num))
                            last_person_proximity_time = current_time

        out.write(annotated)

    cap.release()
    out.release()

    with open(progress_file, "w") as f:
        f.write("100")

    with open(csv_path, mode="w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Event ID", "Frame", "Behavior", "Class", "Distance (m)", "Timestamp (s)", "Event Time (system)", "Closest Person Distance (m)"])
        for entry in csv_log_buffer:
            csv_writer.writerow(entry)

    with open(alert_log_path, mode="w", newline="") as alert_file:
        alert_writer = csv.writer(alert_file)
        alert_writer.writerow(["Alert Time", "Title", "Message"])
        for entry in alert_log_buffer:
            alert_writer.writerow(entry)

    return os.path.basename(output_path)
