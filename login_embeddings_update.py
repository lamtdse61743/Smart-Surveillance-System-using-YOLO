import os
from PIL import Image
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
import torch
import numpy as np

# === Paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_BASE = os.path.join(BASE_DIR, "home_owner_imgs")
OUTPUT_BASE = os.path.join(BASE_DIR, "login_embeddings")

# === Create output directory ===
os.makedirs(OUTPUT_BASE, exist_ok=True)

# === FaceNet model and transform ===
model = InceptionResnetV1(pretrained='vggface2').eval()

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# === Loop through each person folder ===
for person in os.listdir(INPUT_BASE):
    person_folder = os.path.join(INPUT_BASE, person)
    if not os.path.isdir(person_folder):
        continue

    embeddings = []

    for img_name in os.listdir(person_folder):
        img_path = os.path.join(person_folder, img_name)
        try:
            img = Image.open(img_path).convert("RGB")
            tensor = transform(img).unsqueeze(0)
            with torch.no_grad():
                emb = model(tensor).squeeze(0).numpy()
                embeddings.append(emb)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    # === Save averaged embedding ===
    if embeddings:
        avg_embedding = np.mean(embeddings, axis=0)
        output_path = os.path.join(OUTPUT_BASE, person + ".npy")
        np.save(output_path, avg_embedding)
        print(f"Averaged embedding saved: {output_path}")
