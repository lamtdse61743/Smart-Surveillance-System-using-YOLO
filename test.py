import numpy as np
import os

# Get absolute path to the login_embeddings folder
base_dir = os.path.dirname(os.path.abspath(__file__))
folder = os.path.join(base_dir, "login_embeddings")

# Iterate through all subfolders and .npy files
for root, _, files in os.walk(folder):
    for file in files:
        if file.endswith(".npy"):
            path = os.path.join(root, file)
            emb = np.load(path)
            norm = np.linalg.norm(emb)
            print(f"\n{file} (Norm: {norm:.4f}):\n{np.round(emb.flatten(), 4)}")
