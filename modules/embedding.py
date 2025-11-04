import os
import cv2
import zipfile
import numpy as np
import pickle
from insightface.model_zoo import get_model


class FaceEmbedder:
    """
    Robust ArcFace embedding loader.
    Automatically extracts buffalo_l.zip and detects w600k_r50.onnx model.
    Works perfectly in Streamlit Cloud (CPU mode).
    """

    def __init__(self, model_name="buffalo_l", device="cpu"):
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.models_dir = os.path.join(self.project_root, "models")
        self.model_dir = os.path.join(self.models_dir, model_name)
        self.zip_path = os.path.join(self.models_dir, f"{model_name}.zip")

        # ------------------------------------------------------
        # Step 1: Extract zip if model folder missing
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
        # Step 2: Auto-detect ArcFace model file
        # ------------------------------------------------------
        detected_model_dir = None
        for root, dirs, files in os.walk(self.models_dir):
            for f in files:
                # Look for main ArcFace model (w600k_r50.onnx)
                if f.lower() == "w600k_r50.onnx":
                    detected_model_dir = root
                    print(f"🔍 Found ArcFace model: {os.path.join(root, f)}")
                    break
            if detected_model_dir:
                break

        if not detected_model_dir:
            # Log contents for debugging
            print("🧾 Listing files under /models after extraction:")
            for root, dirs, files in os.walk(self.models_dir):
                print("📂", root, "->", files)
            raise FileNotFoundError("❌ Could not locate w600k_r50.onnx after extraction!")

        self.model_dir = detected_model_dir

        # ------------------------------------------------------
        # Step 3: Load ArcFace model (CPU)
        # ------------------------------------------------------
        print(f"⚙️  Loading ArcFace model from {self.model_dir} ...")
        self.model = get_model(self.model_dir)
        if self.model is None:
            raise RuntimeError(f"❌ Could not load ArcFace model from {self.model_dir}")

        self.model.prepare(ctx_id=0, provider="CPUExecutionProvider")
        print(f"✅ ArcFace model ready (device={device})")

    # ------------------------------------------------------------------
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

                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

                img = cv2.resize(img, (112, 112))

                try:
                    emb = self.model.get_feat(img)
                    if emb is not None and emb.shape[-1] == 512:
                        embeddings.append(emb)
                except Exception as e:
                    print(f"⚠️ Failed to embed {file}: {e}")

            if embeddings:
                db[folder] = np.mean(embeddings, axis=0).astype("float32")
                print(f"✅ Processed {folder} ({len(embeddings)} images)")
            else:
                print(f"⚠️ No valid embeddings in {folder}")

        with open(save_path, "wb") as f:
            pickle.dump(db, f)

        print(f"💾 Embeddings saved to {save_path}")
        return db

    # ------------------------------------------------------------------
    def embed_single_image(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Image not found: {image_path}")

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        img = cv2.resize(img, (112, 112))
        emb = self.model.get_feat(img)
        return emb.astype("float32")


# ------------------------------------------------------------------
# Local Execution
# ------------------------------------------------------------------
if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    base_path = os.path.join(project_root, "att-database-of-faces")
    save_path = os.path.join(project_root, "data", "embeddings.pkl")

    embedder = FaceEmbedder(device="cpu")
    embedder.compute_embeddings(base_path, save_path)
