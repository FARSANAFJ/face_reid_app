import os
import cv2
import zipfile
import numpy as np
import pickle
from insightface.model_zoo import get_model


class FaceEmbedder:
    """
    Handles ArcFace embedding extraction.
    Automatically extracts buffalo_l.zip if only the ZIP exists.
    Works for both local and Streamlit deployments (CPU mode).
    """

    def __init__(self, model_name="buffalo_l", device="cpu"):
        # ------------------------------------------------------
        # Paths setup
        # ------------------------------------------------------
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.models_dir = os.path.join(self.project_root, "models")
        self.model_dir = os.path.join(self.models_dir, model_name)
        self.zip_path = os.path.join(self.models_dir, f"{model_name}.zip")

        # ------------------------------------------------------
        # Auto-extract ZIP if needed
        # ------------------------------------------------------
        if not os.path.exists(self.model_dir):
            if os.path.exists(self.zip_path):
                print(f"📦 Extracting model from {self.zip_path} ...")
                with zipfile.ZipFile(self.zip_path, "r") as zip_ref:
                    zip_ref.extractall(self.models_dir)
                print(f"✅ Extracted to {self.models_dir}")
            else:
                raise FileNotFoundError(f"❌ Missing model zip: {self.zip_path}")

        # ------------------------------------------------------
        # Detect correct subfolder (handle nested extraction)
        # ------------------------------------------------------
        if not os.path.exists(self.model_dir):
            # Search for folder containing model.onnx
            for root, dirs, files in os.walk(self.models_dir):
                if "model.onnx" in files:
                    self.model_dir = root
                    print(f"🔧 Adjusted model_dir to {self.model_dir}")
                    break

        if not os.path.exists(self.model_dir):
            raise FileNotFoundError(f"❌ Model folder not found after extraction: {self.model_dir}")

        # ------------------------------------------------------
        # Load ArcFace Model (CPU-only)
        # ------------------------------------------------------
        print(f"⚙️  Loading ArcFace model from {self.model_dir} ...")
        self.model = get_model(self.model_dir)
        if self.model is None:
            raise RuntimeError(f"❌ Could not load ArcFace model from {self.model_dir}")

        self.model.prepare(ctx_id=0, provider="CPUExecutionProvider")
        print(f"✅ ArcFace model ready at {self.model_dir} (device={device})")

    # ------------------------------------------------------------------
    def compute_embeddings(self, base_path, save_path):
        """
        Compute embeddings for all persons in dataset folders.
        Each subfolder represents one identity.
        """
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

                # Ensure 3-channel BGR format
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

                img = cv2.resize(img, (112, 112))

                try:
                    emb = self.model.get_feat(img)
                    if emb is not None and emb.shape[-1] == 512:
                        embeddings.append(emb)
                    else:
                        print(f"⚠️ Invalid embedding shape for {file}")
                except Exception as e:
                    print(f"⚠️ Failed to embed {file}: {e}")

            if embeddings:
                # Average embeddings per identity
                db[folder] = np.mean(embeddings, axis=0).astype("float32")
                print(f"✅ Processed {folder} ({len(embeddings)} images)")
            else:
                print(f"⚠️ No valid embeddings in {folder}")

        # Save embeddings.pkl
        with open(save_path, "wb") as f:
            pickle.dump(db, f)

        print(f"💾 Embeddings saved to {save_path}")
        return db

    # ------------------------------------------------------------------
    def embed_single_image(self, image_path):
        """Compute embedding for a single image (used in Streamlit app)."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Image not found: {image_path}")

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        img = cv2.resize(img, (112, 112))
        emb = self.model.get_feat(img)
        return emb.astype("float32")


# ------------------------------------------------------------------
# Local Execution (for dataset embedding)
# ------------------------------------------------------------------
if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    base_path = os.path.join(project_root, "att-database-of-faces")
    save_path = os.path.join(project_root, "data", "embeddings.pkl")

    embedder = FaceEmbedder(device="cpu")
    embedder.compute_embeddings(base_path, save_path)
