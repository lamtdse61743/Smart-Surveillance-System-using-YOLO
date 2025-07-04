from flask import Flask, render_template, request, jsonify, redirect, url_for
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

app = Flask(__name__)

# === Path Config ===
base_dir = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(base_dir, "uploads")
OUTPUT_DIR = os.path.join(base_dir, "static", "output")
LOG_DIR = os.path.join(base_dir, "log")
LOG_PATH = os.path.join(LOG_DIR, "log.csv")
PROGRESS_FILE = os.path.join(LOG_DIR, "progress.txt")
KNOWN_EMBEDDINGS_DIR = os.path.join(base_dir, "login_embeddings")

# === Ensure Directories Exist ===
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(KNOWN_EMBEDDINGS_DIR, exist_ok=True)

# === Load FaceNet Model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# === Transform for input image ===
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# === Login Page ===
@app.route("/")
def login_page():
    return render_template("login.html")

# === Face Verification Endpoint ===
@app.route("/verify-face", methods=["POST"])
def verify_face():
    file = request.files.get("face_image")
    if not file:
        return jsonify({
            "success": False,
            "message": "No image received",
            "similarity": None
        })

    # Preprocess input face
    image = Image.open(file).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model(img_tensor).cpu().numpy()
    embedding = embedding / np.linalg.norm(embedding)  # Normalize

    matched_name = None
    best_similarity = 0.0
    threshold = 0.6

    # Compare against all known embeddings
    for person_file in os.listdir(KNOWN_EMBEDDINGS_DIR):
        if not person_file.endswith(".npy"):
            continue
        emb_path = os.path.join(KNOWN_EMBEDDINGS_DIR, person_file)
        known_embedding = np.load(emb_path)
        known_embedding = known_embedding / np.linalg.norm(known_embedding)  # Normalize

        sim = cosine_similarity(embedding.reshape(1, -1), known_embedding.reshape(1, -1))[0][0]
        print(f"{person_file} similarity: {sim:.4f}")

        if sim > best_similarity:
            best_similarity = sim
            if sim > threshold:
                matched_name = os.path.splitext(person_file)[0]

    if matched_name:
        return jsonify({
            "success": True,
            "name": matched_name,
            "similarity": round(float(best_similarity), 4)
        })
    else:
        return jsonify({
            "success": False,
            "message": "Face not recognized",
            "similarity": round(float(best_similarity), 4)
        })



# === Upload & Video Processing Page ===
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

        # Run video processing
        process_video(input_path, output_path)

        # Load logs
        log_data = []
        if os.path.exists(LOG_PATH):
            with open(LOG_PATH, newline="") as f:
                reader = csv.reader(f)
                log_data = list(reader)

        return jsonify({
            "output_video": output_filename,
            "log_data": log_data
        })

    return render_template("index.html")

# === Progress Polling ===
@app.route("/progress")
def progress():
    try:
        with open(PROGRESS_FILE, "r") as f:
            percent = f.read().strip()
        return percent
    except:
        return "0"

# === Run App ===
if __name__ == "__main__":
    app.run(debug=True)
