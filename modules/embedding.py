import os
import cv2
import numpy as np
import pickle
import zipfile
from insightface.model_zoo import get_model


class FaceEmbedder:
    """
    Production-ready ArcFace embedding class.
    - Automatically extracts buffalo_l.zip if not unzipped.
    - Handles grayscale or color images.
    - Produces stable 512-D embeddings for Streamlit app.
    """

    def __init__(self, model_name="buffalo_l", device="cpu"):
        # -----------------------------------------------------
        # Model paths
        # -----------------------------------------------------
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.model_dir = os.path.join(base_dir, "models", model_name)
        self.zip_path = os.path.join(base_dir, "models", f"{model_name}.zip")

        # -----------------------------------------------------
        # Auto-extract if not already unzipped
        # -----------------------------------------------------
        if not os.path.exists(self.model_dir):
            if os.path.exists(self.zip_path):
                with zipfile.ZipFile(self.zip_path, "r") as zip_ref:
                    zip_ref.extractall(os.path.join(base_dir, "models"))

        # -----------------------------------------------------
        # Load ArcFace model
        # -----------------------------------------------------
        self.model = get_model(self.model_dir)
        if self.model is None:
            raise RuntimeError(f"Could not load ArcFace model from {self.model_dir}")

        self.model.prepare(ctx_id=0, provider="CPUExecutionProvider")

    # ---------------------------------------------------------
    def compute_embeddings(self, base_path, save_path):
        """Compute mean embeddings per person folder."""
        db = {}
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        for folder in sorted(os.listdir(base_path)):
            person_path = os.path.join(base_path, folder)
            if not os.path.isdir(person_path):
                continue

            embeddings = []
            for file in sorted(os.listdir(person_path)):
                if not file.lower().endswith((".jpg", ".jpeg", ".png", ".pgm")):
                    continue

                img_path = os.path.join(person_path, file)
                img = cv2.imread(img_path)
                if img is None:
                    continue

                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

                img = cv2.resize(img, (112, 112))
                emb = self.model.get_feat(img)
                if emb is not None and emb.shape[-1] == 512:
                    embeddings.append(emb)

            if embeddings:
                db[folder] = np.mean(embeddings, axis=0).astype("float32")

        with open(save_path, "wb") as f:
            pickle.dump(db, f)
        return db

    # ---------------------------------------------------------
    def embed_single_image(self, image_path):
        """Compute embedding for one single image."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Image not found: {image_path}")

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        img = cv2.resize(img, (112, 112))
        emb = self.model.get_feat(img)
        return emb.astype("float32")


# -------------------------------------------------------------
if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    base_path = os.path.join(project_root, "att-database-of-faces")
    save_path = os.path.join(project_root, "data", "embeddings.pkl")

    embedder = FaceEmbedder(device="cpu")
    embedder.compute_embeddings(base_path, save_path)
