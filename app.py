from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os

from video_surveillance_processor import process_video
import threading
app = Flask(__name__)
base_dir = os.path.dirname(os.path.abspath(__file__))
progress_file = os.path.join(base_dir, "log", "progress.txt")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["video"]
        filename = secure_filename(file.filename)
        upload_dir = os.path.join(base_dir, "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        input_path = os.path.join(upload_dir, filename)
        file.save(input_path)

        name_wo_ext = os.path.splitext(filename)[0]
        output_filename = f"processed_{name_wo_ext}.mp4"
        output_dir = os.path.join(base_dir, "static", "output")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)

        process_video(input_path, output_path)
        return jsonify({"output_video": output_filename})

    return render_template("index.html")



@app.route("/progress")
def progress():
    try:
        with open(progress_file, "r") as f:
            percent = f.read().strip()
        return percent
    except:
        return "0"


if __name__ == "__main__":
    app.run(debug=True)
