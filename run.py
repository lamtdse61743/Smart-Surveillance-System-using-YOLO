# --- Imports ---
import cv2
import os
import time
import csv
import platform
import subprocess
from datetime import datetime, timedelta
from collections import defaultdict, deque
import torch
import numpy as np
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
import mediapipe as mp
import uuid

# === Config ===
input_path = "input_videos/face_test5.mp4"
output_path = "output_videos/face_test5_output.mp4"
csv_path = "log/face_test5_log.csv"
alert_log_path = "log/alert_log.csv"

model_general = YOLO("models/yolov9t.pt")
model_box = YOLO("models/box_yolov9t.pt")

focal_px = 700
frame_skip = 5
persistence_duration_sec = 2
alert_cooldown_sec = 10
delivery_suppression_sec = 5
log_cleanup_window_sec = 60
person_proximity_cooldown_sec = 10
mailbox_cooldown_sec = 3600
box_removal_timeout_sec = 60

# Target class mappings and heights in meters
target_classes = {0: "person", 2: "car", 16: "cat", 17: "dog", 80: "box"}
real_height_m = {0: 1.7, 2: 1.4, 16: 0.1, 17: 0.1, 80: 0.1}

# === Face Recognition Setup ===
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
        print(f"[ERROR] Couldn't read {filepath}")
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
        print(f"[ERROR] No face detected in {filepath}")
        return name, np.zeros(512)

known_faces = dict([
    load_known_face("Lam", "home_owner_imgs/Lam/Lam.jpg"),
    load_known_face("William", "home_owner_imgs/William/William.jpeg")
])

# === Prepare output and logging ===
os.makedirs("suspicious", exist_ok=True)
cap = cv2.VideoCapture(input_path)
fps = cap.get(cv2.CAP_PROP_FPS)
fps = 30 if fps == 0 else fps
width, height = int(cap.get(3)), int(cap.get(4))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

csv_log_buffer = []
alert_log_buffer = []

track_history = defaultdict(deque)
distance_history = defaultdict(deque)
min_distance_tracker = {}
last_boxes = {}
behavior_flags = set()
disappeared_tracks = {}
track_timestamps = {}
suspicious_events = []
proximity_flags = set()
box_appearance_nearby = defaultdict(bool)
recent_persons = deque()
logged_boxes = set()
box_last_seen = {}
home_arrivals = set()
last_alert_time = {}
track_to_name = {}
delivery_suppression_until = 0
last_person_proximity_time = 0

# === Utilities ===
def notify_local(title, message, key=None):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    current_time = time.time()

    if title == "Proximity Alert":
        if alert_log_buffer:
            last_entry = alert_log_buffer[-1]
            if last_entry[1] == "Proximity Alert":
                last_msg = last_entry[2]
                if message == last_msg:
                    return False  # Exact message already exists
                try:
                    last_dist = float(last_msg.split("at")[1].split()[0])
                    new_dist = float(message.split("at")[1].split()[0])
                    if abs(last_dist - new_dist) < 0.05:
                        return False  # Skip if difference < 5cm
                except:
                    pass

        # Time-based suppression: ignore if similar alert within 10s
        for entry_time_str, entry_title, _ in reversed(alert_log_buffer[-5:]):
            if entry_title != "Proximity Alert":
                continue
            entry_time = datetime.strptime(entry_time_str, "%Y-%m-%d %H:%M:%S")
            if (datetime.now() - entry_time).total_seconds() < 10:
                return False

    if key and key in last_alert_time and (current_time - last_alert_time[key] < alert_cooldown_sec):
        return False
    last_alert_time[key] = current_time

    alert_log_buffer.append([now, title, message])

    if platform.system() == "Darwin":
        script = f'display notification "{message} at {now}" with title "{title}"'
        subprocess.run(["osascript", "-e", script])
    elif platform.system() == "Windows":
        try:
            from plyer import notification
            notification.notify(title=title, message=f"{message} at {now}", timeout=5)
        except:
            pass
    else:
        print(f"[{title}] {message} at {now}")
    return True



def get_center(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def save_suspicious_clip(start_frame, end_frame, output_filename):
    cap_clip = cv2.VideoCapture(output_path)
    cap_clip.set(cv2.CAP_PROP_POS_FRAMES, max(0, start_frame))
    out_clip = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for _ in range(end_frame - start_frame):
        ret, frame = cap_clip.read()
        if not ret:
            break
        out_clip.write(frame)
    cap_clip.release()
    out_clip.release()

def is_face_in_box(face_bbox, person_box):
    fx, fy, fw, fh = face_bbox
    px1, py1, px2, py2 = person_box
    return px1 <= fx <= px2 and py1 <= fy <= py2 and px1 <= fx + fw <= px2 and py1 <= fy + fh <= py2

def cleanup_logs(person_name, current_time_str):
    current_time = datetime.strptime(current_time_str, "%Y-%m-%d %H:%M:%S")
    cutoff_time = current_time - timedelta(seconds=log_cleanup_window_sec)
    cleaned_alert_buffer = []
    cleaned_csv_buffer = []

    for entry in alert_log_buffer:
        entry_time_str, title, message = entry
        entry_time = datetime.strptime(entry_time_str, "%Y-%m-%d %H:%M:%S")
        if title == "Proximity Alert" and "person approaching" in message.lower() and entry_time >= cutoff_time:
            continue
        cleaned_alert_buffer.append(entry)

    for entry in csv_log_buffer:
        event_id, frame, behavior, cls, distance, ts, event_time, closest_dist = entry
        entry_time = datetime.strptime(event_time, "%Y-%m-%d %H:%M:%S")
        if behavior == "Proximity Alert" and cls == "person" and entry_time >= cutoff_time:
            continue
        cleaned_csv_buffer.append(entry)

    return cleaned_alert_buffer, cleaned_csv_buffer

# === Main Loop ===
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    timestamp = frame_num / fps
    annotated = frame.copy()
    current_ids = set()
    delivery_triggered = False
    current_time = time.time()

    # --- Face Recognition Every 5 Frames ---
    face_results = None
    if frame_num % 5 == 0:
        face_results = mp_face.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if face_results.detections:
            for detection in face_results.detections:
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
                    best_match, best_score = "Unknown", 0.0
                    for name, known_emb in known_faces.items():
                        score = np.dot(emb, known_emb)
                        if score > best_score:
                            best_match, best_score = name, score
                    if best_score > 0.7 and best_match not in home_arrivals:
                        home_arrivals.add(best_match)
                        timestamp_real = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        event_id = str(uuid.uuid4())
                        csv_log_buffer.append([event_id, frame_num, "Door open", best_match, "", round(timestamp, 2), timestamp_real, ""])
                        if notify_local("Home Owner Detected", f"{best_match} just came home!", key=f"face_{best_match}"):
                            print(f"[Frame {frame_num}] {best_match} just came home!")
                            alert_log_buffer.append([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Door Open", f"Owner: {best_match}"])
                        alert_log_buffer[:], csv_log_buffer[:] = cleanup_logs(best_match, timestamp_real)
                        for track_id, ((x1, y1, x2, y2), _, _) in last_boxes.items():
                            if target_classes.get(disappeared_tracks.get(track_id, [None])[0], "") == "person":
                                if is_face_in_box((x, y, w_box, h_box), (x1, y1, x2, y2)):
                                    track_to_name[track_id] = best_match
                                    proximity_flags.discard(track_id)
                                    break
                except Exception as e:
                    print("[Face Error]", e)

    # --- YOLO Detection and Annotation ---
    if frame_num % frame_skip == 0:
        detections = []
        detected_box_centers = set()
        result_general = model_general.track(frame, persist=True, verbose=False, tracker="bytetrack.yaml")[0]
        if result_general.boxes.id is not None:
            for box, cls_id, track_id in zip(result_general.boxes.xyxy, result_general.boxes.cls, result_general.boxes.id):
                cls_id = int(cls_id)
                track_id = int(track_id)
                if cls_id in [0, 2, 16, 17]:
                    detections.append((box, cls_id, track_id))
                    current_ids.add(track_id)

        result_box = model_box(frame, verbose=False)[0]
        for box, cls_id in zip(result_box.boxes.xyxy, result_box.boxes.cls):
            cls_id = int(cls_id)
            if cls_id == 80:
                box_center = get_center(box.tolist())
                detected_box_centers.add(box_center)
                is_new_box = all(
                    abs(box_center[0] - prev_center[0]) > 20 or
                    abs(box_center[1] - prev_center[1]) > 20 or
                    (current_time - prev_time) > mailbox_cooldown_sec
                    for prev_center, prev_time in logged_boxes
                )
                if is_new_box:
                    fake_track_id = hash(str(box_center) + str(current_time)) % (10**6)
                    detections.append((box, cls_id, fake_track_id))
                box_last_seen[box_center] = current_time

        for center in list(box_last_seen.keys()):
            if current_time - box_last_seen[center] > box_removal_timeout_sec:
                logged_boxes.discard(next((c, t) for c, t in logged_boxes if c == center))
                del box_last_seen[center]

        for box, cls_id, track_id in detections:
            label = target_classes.get(cls_id, str(cls_id))
            x1, y1, x2, y2 = map(int, box.tolist())
            center = get_center((x1, y1, x2, y2))
            box_height = y2 - y1
            height_m = real_height_m.get(cls_id, 1.0)
            distance_m = (focal_px * height_m) / box_height if box_height > 0 else None
            distance_text = f"{label}: {distance_m:.2f} m" if distance_m else f"{label}: N/A"

            if distance_m and frame_num <= delivery_suppression_until:
                delivery_triggered = True

            if distance_m and not delivery_triggered:
                if cls_id == 80 and distance_m < 5.0:
                    is_within_cooldown = any(
                        abs(center[0] - prev_center[0]) <= 20 and
                        abs(center[1] - prev_center[1]) <= 20 and
                        (current_time - prev_time) <= mailbox_cooldown_sec
                        for prev_center, prev_time in logged_boxes
                    )
                    if not is_within_cooldown:
                        logged_boxes.add((center, current_time))
                        event_time = round(timestamp, 2)
                        real_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        event_id = str(uuid.uuid4())
                        csv_log_buffer.append([event_id, frame_num, "Box Appeared", label, round(distance_m, 2), event_time, real_time, ""])
                        if notify_local("Delivery Alert", "A mailbox appeared close by.", key=f"box_{track_id}"):
                            delivery_suppression_until = frame_num + int(fps * delivery_suppression_sec)
                            delivery_triggered = True

                if cls_id in [0, 2, 16, 17] and distance_m < 5.0:
                    if cls_id == 0 and track_id in track_to_name:
                        continue
                    if cls_id == 0 and (current_time - last_person_proximity_time < person_proximity_cooldown_sec):
                        continue
                    if track_id not in proximity_flags:
                        if notify_local("Proximity Alert", f"There's a {label} approaching at {round(distance_m, 2)} meters.", key=f"prox_{track_id}"):
                            proximity_flags.add(track_id)
                            if cls_id == 0:
                                recent_persons.append((track_id, distance_m, frame_num))
                                last_person_proximity_time = current_time

                if cls_id == 80:
                    for pid, dist_p, last_seen in recent_persons:
                        if dist_p < 5.0 and (frame_num - last_seen) <= int(fps * persistence_duration_sec):
                            if pid in track_to_name:
                                continue
                            if notify_local("Delivery Detected", "There is a mailbox in front, go check that out!", key=f"delivery_{pid}"):
                                event_id = str(uuid.uuid4())
                                real_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                csv_log_buffer.append([event_id, frame_num, "Delivery Detected", "person+box", "", round(timestamp, 2), real_time, round(dist_p, 2)])
                                recent_persons.clear()
                                proximity_flags.clear()
                                delivery_triggered = True
                                delivery_suppression_until = frame_num + int(fps * delivery_suppression_sec)
                                break

            track_history[track_id].append(center)
            if len(track_history[track_id]) > int(fps * 30):
                track_history[track_id].popleft()

            distance_history[track_id].append(distance_m)
            if len(distance_history[track_id]) > 5:
                distance_history[track_id].popleft()

            min_distance = min(min_distance_tracker.get(track_id, distance_m or float('inf')), distance_m or float('inf'))
            min_distance_tracker[track_id] = min_distance
            if track_id not in track_timestamps:
                track_timestamps[track_id] = [frame_num, frame_num]
            else:
                track_timestamps[track_id][1] = frame_num

            last_boxes[track_id] = ((x1, y1, x2, y2), distance_text, frame_num)
            disappeared_tracks[track_id] = (label, min_distance, frame_num)

    for track_id in list(last_boxes.keys()):
        if track_id not in current_ids:
            last_seen_frame = last_boxes[track_id][2]
            if frame_num - last_seen_frame > int(persistence_duration_sec * fps):
                if track_id in track_to_name:
                    del track_to_name[track_id]
                del last_boxes[track_id]
                if track_id in disappeared_tracks:
                    del disappeared_tracks[track_id]
                if track_id in proximity_flags:
                    proximity_flags.remove(track_id)
                continue

        box, text, last_seen_frame = last_boxes[track_id]
        x1, y1, x2, y2 = box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if face_results and face_results.detections:
        for detection in face_results.detections:
            bbox = detection.location_data.relative_bounding_box
            h, w = frame.shape[:2]
            x, y = int(bbox.xmin * w), int(bbox.ymin * h)
            w_box = int(bbox.width * w)
            h_box = int(bbox.height * h)
            cv2.rectangle(annotated, (x, y), (x + w_box, y + h_box), (0, 0, 255), 2)
            cv2.putText(annotated, best_match if best_score > 0.7 else "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    out.write(annotated)
    cv2.imshow("Detection + Face Recognition", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Write logs to files
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

cap.release()
out.release()
cv2.destroyAllWindows()

print("Saving suspicious video...")
for clip_start, clip_end, clip_time_str in suspicious_events:
    clip_filename = f"suspicious/{clip_time_str}.mp4"
    save_suspicious_clip(clip_start, clip_end, clip_filename)
print("All suspicious videos saved.")