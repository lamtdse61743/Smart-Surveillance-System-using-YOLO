
from flask import Flask, render_template, request, jsonify, redirect, url_for, send_from_directory, Response
from werkzeug.utils import secure_filename
import os
import cv2
import csv
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity
from video_surveillance_processor import process_video, process_frame
import subprocess
import threading
import time
from collections import deque
from flask_socketio import SocketIO
from filelock import FileLock

app = Flask(__name__, static_url_path="/static", static_folder="static")
socketio = SocketIO(app)

# Path Config
base_dir = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(base_dir, "Uploads")
OUTPUT_DIR = os.path.join(base_dir, "static", "output")
HLS_DIR = os.path.join(OUTPUT_DIR, "hls")
LOG_DIR = os.path.join(base_dir, "log")
LOG_PATH = os.path.join(LOG_DIR, "log.csv")
PROGRESS_FILE = os.path.join(LOG_DIR, "progress.txt")
KNOWN_EMBEDDINGS_DIR = os.path.join(base_dir, "login_embeddings")

for folder in [UPLOAD_DIR, OUTPUT_DIR, HLS_DIR, LOG_DIR, KNOWN_EMBEDDINGS_DIR]:
    os.makedirs(folder, exist_ok=True)

# FaceNet Model for Login
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# File Validation
ALLOWED_EXTENSIONS = {".mp4", ".m4v"}
def allowed_file(filename):
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS

# Login Routes
@app.route("/")
def login_page():
    return render_template("login.html")

@app.route("/verify-face", methods=["POST"])
def verify_face():
    file = request.files.get("face_image")
    if not file:
        return jsonify(success=False, message="No image received", similarity=None, name=None)

    try:
        image = Image.open(file).convert("RGB")
        img_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = model(img_tensor).cpu().numpy()
        embedding = embedding / np.linalg.norm(embedding)

        matched_name = None
        best_similarity = 0.0
        threshold = 0.55

        for person_name in os.listdir(KNOWN_EMBEDDINGS_DIR):
            person_dir = os.path.join(KNOWN_EMBEDDINGS_DIR, person_name)
            if not os.path.isdir(person_dir):
                continue

            embeddings = [
                np.load(os.path.join(person_dir, f))
                for f in os.listdir(person_dir) if f.endswith(".npy")
            ]

            if not embeddings:
                continue

            avg_emb = np.mean([e / np.linalg.norm(e) for e in embeddings], axis=0)
            sim = cosine_similarity(embedding.reshape(1, -1), avg_emb.reshape(1, -1))[0][0]

            print(f"{person_name} similarity: {sim:.4f}")

            if sim > best_similarity:
                best_similarity = sim
                if sim > threshold:
                    matched_name = person_name

        similarity_score = round(float(best_similarity), 4)

        if matched_name:
            return jsonify(success=True, name=matched_name, similarity=similarity_score)
        else:
            return jsonify(success=False, message="Face not recognized", similarity=similarity_score, name=None)

    except Exception as e:
        print("Error in verify_face:", str(e))
        return jsonify(success=False, message="Server error during face verification", similarity=None, name=None)

# Video Upload and Processing
@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        file = request.files["video"]
        if not allowed_file(file.filename):
            return jsonify(success=False, message="Invalid video format"), 400
        filename = secure_filename(file.filename)
        input_path = os.path.join(UPLOAD_DIR, filename)
        file.save(input_path)

        name_wo_ext = os.path.splitext(filename)[0]
        output_filename = f"processed_{name_wo_ext}.mp4"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        process_video(input_path, output_path)

        log_data = []
        if os.path.exists(LOG_PATH):
            with FileLock(LOG_PATH + ".lock"):
                with open(LOG_PATH, newline="") as f:
                    reader = csv.reader(f)
                    log_data = list(reader)

        return jsonify(output_video=output_filename, log_data=log_data)

    return render_template("index.html")

@app.route("/progress")
def progress():
    try:
        with FileLock(PROGRESS_FILE + ".lock"):
            with open(PROGRESS_FILE, "r") as f:
                return f.read().strip()
    except:
        return "0"


# Streaming Routes
@app.route("/stream")
def stream_page():
    return render_template("stream.html")


# Serve the log.csv as JSON with optional filtering
@app.route("/log-json")
def log_json():
    # Get filter params
    event_type = request.args.get("type", None)
    person = request.args.get("person", None)
    date_from = request.args.get("date_from", None)
    date_to = request.args.get("date_to", None)
    try:
        if os.path.exists(LOG_PATH):
            with FileLock(LOG_PATH + ".lock"):
                with open(LOG_PATH, newline="") as f:
                    reader = csv.reader(f)
                    rows = list(reader)
            if not rows:
                return jsonify(log_data=[])
            header = rows[0]
            data = rows[1:]
            # Filtering
            filtered = []
            for row in data:
                # Columns: Event ID, Frame, Behavior, Class, Distance (m), Timestamp (s), Event Time (system), Notification
                match = True
                if event_type and event_type.lower() not in row[2].lower():
                    match = False
                if person and person.lower() not in row[3].lower():
                    match = False
                if date_from or date_to:
                    # Event Time (system) is row[6], format: YYYY-MM-DD HH:MM:SS
                    try:
                        event_time = row[6][:19]
                        if date_from and event_time < date_from:
                            match = False
                        if date_to and event_time > date_to:
                            match = False
                    except:
                        pass
                if match:
                    filtered.append(row)
            return jsonify(log_data=[header] + filtered)
        else:
            return jsonify(log_data=[])
    except Exception as e:
        return jsonify(log_data=[], error=str(e)), 500

# Route for event statistics (for charts)
@app.route("/log-stats")
def log_stats():
    try:
        if not os.path.exists(LOG_PATH):
            print("[log-stats] log.csv does not exist")
            return jsonify(stats={})
        with FileLock(LOG_PATH + ".lock"):
            with open(LOG_PATH, newline="") as f:
                reader = csv.reader(f)
                rows = list(reader)
        print(f"[log-stats] header: {rows[0] if rows else 'NO HEADER'}")
        print(f"[log-stats] first 3 rows: {rows[1:4] if len(rows) > 1 else 'NO DATA'}")
        if len(rows) < 2:
            print("[log-stats] Not enough rows for stats")
            return jsonify(stats={})
        header = rows[0]
        data = rows[1:]
        # Defensive: skip rows with wrong length
        data = [row for row in data if len(row) == len(header)]
        print(f"[log-stats] filtered data rows: {len(data)}")
        # Count by event type (Behavior)
        from collections import Counter, defaultdict
        type_counts = Counter(row[2] for row in data)
        person_counts = Counter(row[3] for row in data)
        hour_counts = defaultdict(int)
        for row in data:
            try:
                hour = row[6][11:13]  # HH from YYYY-MM-DD HH:MM:SS
                hour_counts[hour] += 1
            except Exception as e:
                print(f"[log-stats] hour parse error: {e} row={row}")
        stats = {
            "type_counts": dict(type_counts),
            "person_counts": dict(person_counts),
            "hour_counts": dict(hour_counts)
        }
        print(f"[log-stats] type_counts: {stats['type_counts']}")
        print(f"[log-stats] person_counts: {stats['person_counts']}")
        print(f"[log-stats] hour_counts: {stats['hour_counts']}")
        return jsonify(stats=stats)
    except Exception as e:
        print(f"[log-stats] error: {e}")
        return jsonify(stats={}, error=str(e)), 500

# Gemini Chat Page (GET: render, POST: chat)
@app.route("/chat", methods=["GET", "POST"])
def chat_page():
    if request.method == "GET":
        return render_template("chat.html")
    # POST: handle chat
    data = request.get_json()
    user_message = data.get("message", "")
    # --- RAG: Retrieve relevant log context ---
    log_context = ""
    try:
        if os.path.exists(LOG_PATH):
            with FileLock(LOG_PATH + ".lock"):
                with open(LOG_PATH, newline="") as f:
                    rows = list(csv.reader(f))
                    # Get last 20 events for context
                    if len(rows) > 1:
                        header = rows[0]
                        last_events = rows[-20:]
                        log_context = "\n".join([", ".join(row) for row in last_events])
    except Exception as e:
        log_context = "(Could not load log context)"

    # --- Gemini API call (placeholder) ---
    # You must set your Gemini API key as GEMINI_API_KEY env var
    import requests
    GEMINI_API_KEY = "AIzaSyCGM9wvYHrU3F6qhkBfF3wjMVktmqKt_tY"
    print(f"[Gemini] Using API key: {GEMINI_API_KEY}")
    gemini_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=" + GEMINI_API_KEY
    prompt = f"You are the smart surveillance system for this house. Here is a summary of recent events from the surveillance log:\n{log_context}\n\nUser question: {user_message}\n\nPlease answer in a friendly, conversational way, as if you are the home's helpful AI guardian."
    reply = "(Gemini API not configured)"
    if GEMINI_API_KEY:
        try:
            payload = {
                "contents": [{"parts": [{"text": prompt}]}]
            }
            print(f"[Gemini] Sending request to: {gemini_url}")
            resp = requests.post(gemini_url, json=payload, timeout=40)
            print(f"[Gemini] Response status: {resp.status_code}")
            if resp.ok:
                data = resp.json()
                print(f"[Gemini] Response data: {data}")
                reply = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "(No answer)")
            else:
                reply = f"Gemini API error: {resp.status_code}"
        except Exception as e:
            print(f"[Gemini] Exception: {e}")
            reply = f"Gemini API error: {str(e)}"
    return jsonify(reply=reply)

def run_stream_processing(input_video_path, hls_output_path):
    # Initialize video capture
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {input_video_path}")
        socketio.emit("stream_stopped")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # FFmpeg command to pipe processed frames and input audio
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",  # raw video from stdin  
        "-f", "lavfi",
        "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",  # fake audio input
        "-shortest",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-pix_fmt", "yuv420p",
        "-profile:v", "baseline",
        "-level", "3.0",
        "-g", "30",
        "-sc_threshold", "0",
        "-c:a", "aac",
        "-b:a", "128k",
        "-ar", "44100",
        "-ac", "2",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-f", "hls",
        "-hls_time", "4",
        "-hls_list_size", "1000",
        "-hls_flags", "delete_segments+append_list+program_date_time",
        "-hls_segment_filename", os.path.join(hls_output_path, "stream%d.ts"),
        os.path.join(hls_output_path, "stream.m3u8")
    ]


    # Start FFmpeg process in binary mode
    ffmpeg_process = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=False  # Binary mode for stdin
    )

    # Log FFmpeg output
    def stream_ffmpeg_output():
        for line in ffmpeg_process.stderr:
            print(f"FFmpeg: {line.decode('utf-8', errors='ignore').strip()}")
    threading.Thread(target=stream_ffmpeg_output, daemon=True).start()

    # Continuous frame processing
    log_buffer = []
    recent_persons = deque()
    last_person_proximity_time = [0]
    home_arrivals = set()
    persistent_boxes = []
    frame_num = 0

    # Write initial log header if file doesn't exist
    if not os.path.exists(LOG_PATH):
        with FileLock(LOG_PATH + ".lock"):
            with open(LOG_PATH, mode="w", newline="") as log_file:
                log_writer = csv.writer(log_file)
                log_writer.writerow(["Event ID", "Frame", "Behavior", "Class", "Distance (m)", "Timestamp (s)", "Event Time (system)", "Notification"])

    # Wait briefly to ensure FFmpeg is ready
    time.sleep(1)

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset for looping
        for _ in range(frame_count):
            try:
                ret, frame = cap.read()
                if not ret:
                    print("End of video reached, restarting loop")
                    break
                frame_num += 1
                annotated, log_buffer, recent_persons, last_person_proximity_time, home_arrivals, persistent_boxes = process_frame(
                    frame, frame_num, fps, log_buffer, recent_persons, last_person_proximity_time, home_arrivals, persistent_boxes,
                    emit_callback=lambda entry: socketio.emit("log_update", {"row": entry})
                )
                # Write frame to FFmpeg pipe
                ffmpeg_process.stdin.write(annotated.tobytes())
                ffmpeg_process.stdin.flush()
                # Append new log entries to file
                if log_buffer:
                    with FileLock(LOG_PATH + ".lock"):
                        with open(LOG_PATH, mode="a", newline="") as log_file:
                            log_writer = csv.writer(log_file)
                            for entry in log_buffer:
                                log_writer.writerow(entry)
                            log_file.flush()
                            os.fsync(log_file.fileno())
                    log_buffer.clear()
                time.sleep(1 / fps)  # Simulate real-time
            except Exception as e:
                print(f"Frame processing error: {e}")
                break
        if ffmpeg_process.poll() is not None:  # FFmpeg terminated
            print("FFmpeg process ended unexpectedly")
            break

    cap.release()
    ffmpeg_process.stdin.close()
    ffmpeg_process.terminate()
    try:
        ffmpeg_process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        ffmpeg_process.kill()
    socketio.emit("stream_stopped")

@app.route("/upload-stream", methods=["POST"])
def upload_stream():
    file = request.files.get("video")
    if not file or not allowed_file(file.filename):
        return jsonify(success=False, message="Invalid or no video"), 400

    video_path = os.path.join(UPLOAD_DIR, "stream_video.mp4")
    for f in os.listdir(HLS_DIR):
        os.remove(os.path.join(HLS_DIR, f))
    file.save(video_path)

    with FileLock(PROGRESS_FILE + ".lock"):
        with open(PROGRESS_FILE, "w") as f:
            f.write("0")

    thread = threading.Thread(target=run_stream_processing, args=(video_path, HLS_DIR))
    thread.daemon = True
    thread.start()

    return jsonify(success=True), 200

@app.route("/stream-status")
def stream_status():
    playlist = os.path.join(HLS_DIR, "stream.m3u8")
    return jsonify(live=os.path.exists(playlist))

@app.route("/hls/<path:filename>")
def hls_stream(filename):
    return send_from_directory(HLS_DIR, filename)

@app.route("/stop-stream", methods=["POST"])
def stop_stream():
    for f in os.listdir(HLS_DIR):
        os.remove(os.path.join(HLS_DIR, f))
    socketio.emit("stream_stopped")
    return jsonify(success=True)

if __name__ == "__main__":
    socketio.run(app, debug=True)
