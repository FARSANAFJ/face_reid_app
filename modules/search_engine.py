import os
import pickle
import numpy as np
import faiss


class FaceSearchEngine:
    """
    Loads precomputed embeddings and performs fast similarity search
    using FAISS (cosine similarity). Works with webcam & uploaded images.
    """

    def __init__(self):
        # Locate project root
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.embedding_path = os.path.join(self.project_root, "data", "embeddings.pkl")
        self.index_path = os.path.join(self.project_root, "data", "faiss_index.bin")

        self.embeddings = None
        self.labels = None
        self.index = None

        # Try to load embeddings & FAISS index
        try:
            self.load_embeddings()
            self.load_or_build_index()
        except Exception as e:
            print(f"⚠️ Initialization warning: {e}")

    # -------------------------------------------------------------
    def load_embeddings(self):
        """Load precomputed embeddings."""
        if not os.path.exists(self.embedding_path):
            raise FileNotFoundError(f"❌ Embedding file not found: {self.embedding_path}")

        with open(self.embedding_path, "rb") as f:
            db = pickle.load(f)

        if not db:
            raise ValueError("❌ Embedding file is empty. Run embedding.py first.")

        self.labels = list(db.keys())
        self.embeddings = np.stack(list(db.values())).astype("float32")

        print(f"✅ Loaded {len(self.labels)} embeddings from {self.embedding_path}")

    # -------------------------------------------------------------
    def load_or_build_index(self):
        """Load FAISS index if it exists, otherwise build one."""
        if os.path.exists(self.index_path):
            try:
                self.index = faiss.read_index(self.index_path)
                print(f"📂 Loaded existing FAISS index from {self.index_path}")
                return
            except Exception as e:
                print(f"⚠️ Failed to load existing index: {e}. Rebuilding...")
        self.build_index()

    # -------------------------------------------------------------
    def build_index(self):
        """Build FAISS index using cosine similarity."""
        if self.embeddings is None or len(self.embeddings) == 0:
            raise ValueError("❌ No embeddings available to build index.")

        self.embeddings = np.squeeze(self.embeddings)
        if self.embeddings.ndim == 1:
            self.embeddings = self.embeddings.reshape(1, -1)

        faiss.normalize_L2(self.embeddings)
        dim = self.embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dim)  # Inner product = cosine similarity
        self.index.add(self.embeddings)
        faiss.write_index(self.index, self.index_path)

        print(f"💾 FAISS cosine-similarity index built and saved to {self.index_path}")

    # -------------------------------------------------------------
    def search(self, new_embedding, threshold=0.35, top_k=3):
        """
        Search for the most similar embedding(s) using cosine similarity.
        Works with embeddings from webcam or uploaded images.
        Returns a sorted list of (label, similarity) tuples.
        """
        if self.index is None:
            raise RuntimeError("FAISS index not built or loaded.")

        # Normalize input embedding
        new_embedding = np.array(new_embedding, dtype="float32")
        new_embedding = np.squeeze(new_embedding)

        if new_embedding.ndim == 1:
            new_embedding = new_embedding.reshape(1, -1)
        elif new_embedding.ndim > 2:
            new_embedding = new_embedding.reshape(1, new_embedding.shape[-1])

        faiss.normalize_L2(new_embedding)

        # Perform top-k similarity search
        D, I = self.index.search(new_embedding, top_k)

        results = []
        for sim, idx in zip(D[0], I[0]):
            if 0 <= idx < len(self.labels):
                label = self.labels[idx]
                results.append((label, float(sim)))

        # Sort by similarity
        results = sorted(results, key=lambda x: x[1], reverse=True)

        # Print the top match to console (debug-friendly)
        if results:
            top_label, top_score = results[0]
            if top_score > threshold:
                print(f"✅ Match found: {top_label} (similarity={top_score:.4f})")
            else:
                print(f"🆕 No confident match (best similarity={top_score:.4f})")
        else:
            print("⚠️ No results returned from FAISS search.")

        return results
