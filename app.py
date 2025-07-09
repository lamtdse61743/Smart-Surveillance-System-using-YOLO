from flask import Flask, render_template, request, jsonify, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import os
import csv
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity
from video_surveillance_processor import process_video
import subprocess

app = Flask(__name__, static_url_path="/static", static_folder="static")

# === Path Config ===
base_dir = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(base_dir, "uploads")
OUTPUT_DIR = os.path.join(base_dir, "static", "output")
HLS_DIR = os.path.join(OUTPUT_DIR, "hls")
LOG_DIR = os.path.join(base_dir, "log")
LOG_PATH = os.path.join(LOG_DIR, "log.csv")
PROGRESS_FILE = os.path.join(LOG_DIR, "progress.txt")
KNOWN_EMBEDDINGS_DIR = os.path.join(base_dir, "login_embeddings")

# === Ensure Directories Exist ===
for folder in [UPLOAD_DIR, OUTPUT_DIR, HLS_DIR, LOG_DIR, KNOWN_EMBEDDINGS_DIR]:
    os.makedirs(folder, exist_ok=True)

# === Load FaceNet Model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# === Image Transform ===
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

@app.route("/")
def login_page():
    return render_template("login.html")

@app.route("/verify-face", methods=["POST"])
def verify_face():
    file = request.files.get("face_image")
    if not file:
        return jsonify(success=False, message="No image received", similarity=None)

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

    if matched_name:
        return jsonify(success=True, name=matched_name, similarity=round(float(best_similarity), 4))
    else:
        return jsonify(success=False, message="Face not recognized", similarity=round(best_similarity, 4))

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        file = request.files["video"]
        filename = secure_filename(file.filename)
        input_path = os.path.join(UPLOAD_DIR, filename)
        file.save(input_path)

        name_wo_ext = os.path.splitext(filename)[0]
        output_filename = f"processed_{name_wo_ext}.mp4"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        process_video(input_path, output_path)

        log_data = []
        if os.path.exists(LOG_PATH):
            with open(LOG_PATH, newline="") as f:
                reader = csv.reader(f)
                log_data = list(reader)

        return jsonify(output_video=output_filename, log_data=log_data)

    return render_template("index.html")

@app.route("/progress")
def progress():
    try:
        with open(PROGRESS_FILE, "r") as f:
            return f.read().strip()
    except:
        return "0"

@app.route("/stream")
def stream_page():
    return render_template("stream.html")

@app.route("/upload-stream", methods=["POST"])
def upload_stream():
    file = request.files.get("video")
    if not file:
        return "No video", 400

    video_path = os.path.join(UPLOAD_DIR, "stream_video.mp4")
    hls_folder = os.path.join(OUTPUT_DIR, "hls")
    
    # Clean old HLS files
    for f in os.listdir(hls_folder):
        os.remove(os.path.join(hls_folder, f))

    file.save(video_path)

    # Run ffmpeg with livestream-like flags
    subprocess.Popen([
        "ffmpeg",
        "-re",  # Stream in real-time
        "-i", video_path,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-g", "60",
        "-sc_threshold", "0",
        "-c:a", "aac",
        "-ar", "44100",
        "-b:a", "128k",
        "-f", "hls",
        "-hls_time", "2",
        "-hls_list_size", "5",  # Keep only last 5 segments
        "-hls_flags", "delete_segments+omit_endlist",  # Don't show future segments
        "-hls_segment_filename", os.path.join(hls_folder, "stream%d.ts"),
        os.path.join(hls_folder, "stream.m3u8")
    ])

    return "", 204

@app.route("/hls/<path:filename>")
def hls_stream(filename):
    return send_from_directory(HLS_DIR, filename)

# === Run App ===
if __name__ == "__main__":
    app.run(debug=True)
