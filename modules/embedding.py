import os
import cv2
import numpy as np
import pickle
from insightface.model_zoo import get_model


class FaceEmbedder:
    """
    Directly computes ArcFace embeddings (no detection).
    Perfect for cropped grayscale datasets like ORL (AT&T).
    """

    def __init__(self, model_name="buffalo_l", device="cpu"):
        self.model = get_model(model_name)
        self.model.prepare(ctx_id=0, provider='CPUExecutionProvider')
        print(f"✅ ArcFace recognizer '{model_name}' initialized on {device.upper()}")

    # -------------------------------------------------------------
    def compute_embeddings(self, base_path, save_path):
        print(f"📂 Reading dataset from: {base_path}")
        db = {}
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        folders = sorted(os.listdir(base_path))
        print(f"🔍 Found {len(folders)} folders.")

        for folder in folders:
            person_path = os.path.join(base_path, folder)
            if not os.path.isdir(person_path):
                continue

            embeddings = []
            print(f"🧠 Processing folder: {folder}")

            for file in sorted(os.listdir(person_path)):
                if not file.lower().endswith(('.jpg', '.jpeg', '.png', '.pgm')):
                    continue

                img_path = os.path.join(person_path, file)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"⚠️ Skipped unreadable image: {img_path}")
                    continue

                # Convert grayscale → BGR
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

                # Resize to ArcFace input size (112x112)
                img = cv2.resize(img, (112, 112))

                # ✅ Correct call for your version
                emb = self.model.get_feat(img)
                embeddings.append(emb)

            if embeddings:
                db[folder] = np.mean(embeddings, axis=0)
                print(f"✅ Processed {folder} ({len(embeddings)} images)")
            else:
                print(f"⚠️ No valid embeddings in {folder}")

        # Save all embeddings
        with open(save_path, "wb") as f:
            pickle.dump(db, f)

        print(f"💾 Embeddings saved to {save_path}")
        return db

    # -------------------------------------------------------------
    def embed_single_image(self, image_path):
        """Compute embedding for one image."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Image not found: {image_path}")

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        img = cv2.resize(img, (112, 112))
        emb = self.model.get_feat(img)
        print(f"✅ Embedding computed for {os.path.basename(image_path)}")
        return emb


# -------------------------------------------------------------
# MAIN EXECUTION
# -------------------------------------------------------------
if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    base_path = os.path.join(project_root, "att-database-of-faces")
    save_path = os.path.join(project_root, "data", "embeddings.pkl")

    print(f"🚀 Project root: {project_root}")
    print(f"📂 Dataset path: {base_path}")
    print(f"📁 Save path: {save_path}")

    embedder = FaceEmbedder(device="cpu")
    embedder.compute_embeddings(base_path, save_path)
