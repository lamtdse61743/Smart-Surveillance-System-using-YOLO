from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import csv

from video_surveillance_processor import process_video

app = Flask(__name__)
base_dir = os.path.dirname(os.path.abspath(__file__))

UPLOAD_DIR = os.path.join(base_dir, "uploads")
OUTPUT_DIR = os.path.join(base_dir, "static", "output")
LOG_PATH = os.path.join(base_dir, "log", "log.csv")
PROGRESS_FILE = os.path.join(base_dir, "log", "progress.txt")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(base_dir, "log"), exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["video"]
        filename = secure_filename(file.filename)
        input_path = os.path.join(UPLOAD_DIR, filename)
        file.save(input_path)

        name_wo_ext = os.path.splitext(filename)[0]
        output_filename = f"processed_{name_wo_ext}.mp4"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        # Process video and generate log
        process_video(input_path, output_path)

        # Read log data
        log_data = []
        if os.path.exists(LOG_PATH):
            with open(LOG_PATH, newline="") as csvfile:
                reader = csv.reader(csvfile)
                log_data = list(reader)

        return jsonify({
            "output_video": output_filename,
            "log_data": log_data
        })

    return render_template("index.html")

@app.route("/progress")
def progress():
    try:
        with open(PROGRESS_FILE, "r") as f:
            percent = f.read().strip()
        return percent
    except:
        return "0"

if __name__ == "__main__":
    app.run(debug=True)
