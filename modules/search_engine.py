import os
import pickle
import numpy as np
import faiss


class FaceSearchEngine:
    """
    Face similarity engine using FAISS with cosine similarity.
    Loads precomputed embeddings and allows efficient face matching.
    """

    def __init__(self):
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.embedding_path = os.path.join(self.project_root, "data", "embeddings.pkl")
        self.index_path = os.path.join(self.project_root, "data", "faiss_index.bin")

        self.embeddings = None
        self.labels = None
        self.index = None

        try:
            self.load_embeddings()
            self.load_or_build_index()
        except Exception as e:
            # Avoid cluttering Streamlit logs
            print(f"[WARN] Initialization issue: {e}")

    # -------------------------------------------------------------
    def load_embeddings(self):
        """Load embeddings safely from pickle and normalize shapes."""
        if not os.path.exists(self.embedding_path):
            raise FileNotFoundError(f"Embedding file missing: {self.embedding_path}")

        with open(self.embedding_path, "rb") as f:
            db = pickle.load(f)

        if not db:
            raise ValueError("Embedding file is empty. Run embedding.py first.")

        clean_embs, clean_labels = [], []
        for label, vec in db.items():
            arr = np.array(vec, dtype="float32").squeeze()
            # Ensure valid 512-D embedding
            if arr.ndim == 1 and arr.shape[0] == 512 and not np.isnan(arr).any():
                clean_embs.append(arr)
                clean_labels.append(label)

        if not clean_embs:
            raise ValueError("No valid embeddings found in the database.")

        self.labels = clean_labels
        self.embeddings = np.stack(clean_embs).astype("float32")

    # -------------------------------------------------------------
    def load_or_build_index(self):
        """Load FAISS index if it exists, else rebuild it."""
        if os.path.exists(self.index_path):
            try:
                self.index = faiss.read_index(self.index_path)
                return
            except Exception:
                pass
        self.build_index()

    # -------------------------------------------------------------
    def build_index(self):
        """Build FAISS cosine similarity index."""
        if self.embeddings is None or len(self.embeddings) == 0:
            raise ValueError("No embeddings to build FAISS index.")

        faiss.normalize_L2(self.embeddings)
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)
        faiss.write_index(self.index, self.index_path)

    # -------------------------------------------------------------
    def search(self, new_embedding, threshold=0.35, top_k=3):
        """Find top-K matches using cosine similarity."""
        if self.index is None:
            raise RuntimeError("FAISS index not built or loaded.")

        new_embedding = np.array(new_embedding, dtype="float32").squeeze()

        # Reshape to (1, 512)
        if new_embedding.ndim == 1:
            new_embedding = new_embedding.reshape(1, -1)
        elif new_embedding.ndim > 2:
            new_embedding = new_embedding.reshape(1, new_embedding.shape[-1])

        if new_embedding.shape[1] != 512:
            raise ValueError(f"Invalid embedding shape: {new_embedding.shape}")

        faiss.normalize_L2(new_embedding)

        D, I = self.index.search(new_embedding, top_k)
        results = []

        for sim, idx in zip(D[0], I[0]):
            if 0 <= idx < len(self.labels):
                results.append((self.labels[idx], float(sim)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results
