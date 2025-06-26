from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
from video_surveillance_processor import process_video

app = Flask(__name__)

# Set folders
UPLOAD_FOLDER = 'static/input'
OUTPUT_FOLDER = 'static/output'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    output_video = None

    if request.method == "POST":
        uploaded_file = request.files.get("video")
        if uploaded_file and uploaded_file.filename != "":
            filename = secure_filename(uploaded_file.filename)
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            output_filename = f"processed_{filename}"
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

            # Save uploaded file
            uploaded_file.save(input_path)

            # Process the video using your full surveillance logic
            process_video(input_path, output_path)

            # Show result
            output_video = output_filename

    return render_template("index.html", output_video=output_video)

if __name__ == "__main__":
    app.run(debug=True)
