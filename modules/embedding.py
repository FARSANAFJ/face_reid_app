import os
import cv2
import numpy as np
import pickle
import zipfile
from insightface.model_zoo import get_model


class FaceEmbedder:
    """
    Loads ArcFace model (auto-extracts buffalo_l.zip if needed)
    and computes embeddings for face recognition.
    """

    def __init__(self, model_name="buffalo_l", device="cpu"):
        self.model_name = model_name
        self.device = device

        # Path setup
        self.models_root = os.path.join(os.path.dirname(__file__), "..", "models")
        self.model_dir = os.path.join(self.models_root, self.model_name)
        self.zip_path = os.path.join(self.models_root, f"{self.model_name}.zip")

        # ------------------------------------------------------------
        # 🧩 Auto-unzip model if folder is missing but zip exists
        # ------------------------------------------------------------
        if not os.path.exists(self.model_dir):
            if os.path.exists(self.zip_path):
                print(f"📦 Extracting {self.zip_path}...")
                with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                    zip_ref.extractall(self.models_root)
                print("✅ Model extracted successfully!")
            else:
                raise FileNotFoundError(
                    f"❌ Neither model folder nor zip found: {self.model_dir}"
                )

        # ------------------------------------------------------------
        # Load ArcFace model
        # ------------------------------------------------------------
        self.model = get_model(self.model_dir)
        if self.model is None:
            raise RuntimeError(f"❌ Could not load ArcFace model from {self.model_dir}")

        self.model.prepare(ctx_id=0, provider="CPUExecutionProvider")
        print(f"✅ ArcFace model ready at {self.model_dir} (device={device})")

    # ------------------------------------------------------------
    def compute_embeddings(self, base_path, save_path):
        """Generate embeddings for all faces in dataset folders."""
        db = {}
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        folders = sorted(os.listdir(base_path))

        for folder in folders:
            folder_path = os.path.join(base_path, folder)
            if not os.path.isdir(folder_path):
                continue

            embeddings = []
            for file in sorted(os.listdir(folder_path)):
                if not file.lower().endswith(('.jpg', '.jpeg', '.png', '.pgm')):
                    continue

                img_path = os.path.join(folder_path, file)
                img = cv2.imread(img_path)
                if img is None:
                    continue

                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                img = cv2.resize(img, (112, 112))

                try:
                    emb = self.model.get_feat(img)
                    if emb is not None and emb.shape[-1] == 512:
                        embeddings.append(emb)
                except Exception:
                    pass

            if embeddings:
                db[folder] = np.mean(embeddings, axis=0).astype("float32")

        with open(save_path, "wb") as f:
            pickle.dump(db, f)

        print(f"💾 Saved embeddings to {save_path}")
        return db

    # ------------------------------------------------------------
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
