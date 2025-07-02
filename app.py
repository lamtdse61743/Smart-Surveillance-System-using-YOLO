from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import csv

from video_surveillance_processor import process_video

app = Flask(__name__)

# === Path Config ===
base_dir = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(base_dir, "uploads")
OUTPUT_DIR = os.path.join(base_dir, "static", "output")
LOG_DIR = os.path.join(base_dir, "log")
LOG_PATH = os.path.join(LOG_DIR, "log.csv")
PROGRESS_FILE = os.path.join(LOG_DIR, "progress.txt")

# === Ensure Directories Exist ===
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# === Main Page & Upload Handling ===
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Save uploaded video
        file = request.files["video"]
        filename = secure_filename(file.filename)
        input_path = os.path.join(UPLOAD_DIR, filename)
        file.save(input_path)

        # Generate output filename
        name_wo_ext = os.path.splitext(filename)[0]
        output_filename = f"processed_{name_wo_ext}.mp4"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        # Process the video
        process_video(input_path, output_path)

        # Read log data
        log_data = []
        if os.path.exists(LOG_PATH):
            with open(LOG_PATH, newline="") as f:
                reader = csv.reader(f)
                log_data = list(reader)
        print("Log rows returned:", len(log_data))
        print("Sample log row:", log_data[1] if len(log_data) > 1 else "No data")

        return jsonify({
            "output_video": output_filename,
            "log_data": log_data
        })

    return render_template("index.html")

# === Progress Polling Endpoint ===
@app.route("/progress")
def progress():
    try:
        with open(PROGRESS_FILE, "r") as f:
            percent = f.read().strip()
        return percent
    except:
        return "0"

# === Run the Flask App ===
if __name__ == "__main__":
    app.run(debug=True)
