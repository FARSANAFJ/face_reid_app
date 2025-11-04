import os
import cv2
import numpy as np
import pickle
import streamlit as st
from PIL import Image

# Import local modules
from modules.embedding import FaceEmbedder
from modules.search_engine import FaceSearchEngine

# -------------------------------------------------------------
# Streamlit Configuration
# -------------------------------------------------------------
st.set_page_config(
    page_title="Face Re-Identification",
    page_icon="🧠",
    layout="centered"
)

# -------------------------------------------------------------
# Custom UI Styling
# -------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #edf2fb 0%, #e3f2fd 100%);
        color: #1e293b;
        font-family: 'Segoe UI', sans-serif;
    }
    h1, h2, h3 {
        color: #0f172a;
        font-weight: 600;
    }
    .stButton>button {
        background: linear-gradient(90deg, #2563eb, #06b6d4);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6em 1.2em;
        font-size: 16px;
        font-weight: 600;
        transition: all 0.3s ease-in-out;
        box-shadow: 0 2px 6px rgba(0,0,0,0.15);
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #1d4ed8, #0891b2);
        transform: scale(1.03);
        box-shadow: 0px 0px 10px rgba(37, 99, 235, 0.4);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: rgba(255, 255, 255, 0.7);
        padding: 8px;
        border-radius: 10px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #2563eb, #06b6d4);
        color: white !important;
        font-weight: 600;
        box-shadow: 0px 2px 6px rgba(0,0,0,0.15);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------------------
# App Title
# -------------------------------------------------------------
st.title("🎓 Face Re-Identification Prototype")
st.caption("Capture or upload faces to register or match against stored embeddings.")

# -------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------
def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    """Convert PIL image to OpenCV BGR format."""
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

# -------------------------------------------------------------
# Load Models (cached)
# -------------------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_models():
    embedder = FaceEmbedder(device="cpu")
    engine = FaceSearchEngine()
    return embedder, engine

embedder, engine = load_models()

# -------------------------------------------------------------
# Tabs: Register / Match
# -------------------------------------------------------------
tab1, tab2 = st.tabs(["🧩 Register New Face", "🔍 Match Existing Face"])

# =============================================================
# 🧩 TAB 1 — Register New Face
# =============================================================
with tab1:
    st.header("Register a New Face")

    cam = st.camera_input("Capture your face to register")
    name = st.text_input("Enter a unique name for this person (e.g., minu_01):")

    if st.button("💾 Save Face Embedding", use_container_width=True):
        if cam and name.strip():
            try:
                # Save captured face
                save_dir = os.path.join(engine.project_root, "captured_faces")
                os.makedirs(save_dir, exist_ok=True)
                img_path = os.path.join(save_dir, f"{name}.jpg")
                img = Image.open(cam)
                img.save(img_path)

                # Extract embedding directly (no .app call)
                bgr = pil_to_bgr(img)
                emb = embedder.model.get_feat(cv2.resize(bgr, (112, 112)))
                emb = np.array(emb, dtype="float32").squeeze()

                # Ensure data directory exists before saving embeddings
                db = {}
                data_dir = os.path.dirname(engine.embedding_path)
                os.makedirs(data_dir, exist_ok=True)

                if os.path.exists(engine.embedding_path):
                    with open(engine.embedding_path, "rb") as f:
                        db = pickle.load(f)

                db[name] = emb
                with open(engine.embedding_path, "wb") as f:
                    pickle.dump(db, f)

                # Rebuild FAISS index
                engine.load_embeddings()
                engine.build_index()

                st.success(f"✅ Registered {name} successfully!")
                st.image(img_path, caption=f"Saved as: {name}", use_container_width=True)

            except Exception as e:
                st.error(f"Error during save: {e}")
        else:
            st.warning("Please capture a face and enter a valid name.")

# =============================================================
# 🔍 TAB 2 — Match Existing Face
# =============================================================
with tab2:
    st.header("Match a Captured or Uploaded Face")

    mode = st.radio("Choose input source:", ["📁 Upload image", "📷 Use webcam"], horizontal=True)
    threshold = st.slider("Match threshold (cosine similarity):", 0.1, 0.95, 0.35, 0.01)

    uploaded_img = None
    if mode == "📁 Upload image":
        file = st.file_uploader("Upload face image", type=["jpg", "jpeg", "png", "pgm"])
        if file:
            uploaded_img = Image.open(file)
            st.image(uploaded_img, caption="Uploaded Image", use_container_width=True)
    else:
        cam = st.camera_input("Capture image for matching")
        if cam:
            uploaded_img = Image.open(cam)
            st.image(uploaded_img, caption="Captured Image", use_container_width=True)

    if st.button("🔎 Match Face", use_container_width=True):
        if uploaded_img is None:
            st.warning("Please capture or upload an image first.")
            st.stop()

        try:
            bgr = pil_to_bgr(uploaded_img)
            emb = embedder.model.get_feat(cv2.resize(bgr, (112, 112)))
            emb = np.array(emb, dtype="float32").squeeze()

            results = engine.search(emb, threshold=threshold, top_k=3)

            if not results:
                st.error("⚠️ No embeddings found or no match detected.")
                st.stop()

            best_label, best_score = results[0]
            if best_score > threshold:
                st.success(f"✅ Match Found: **{best_label}**")
                st.metric("Similarity", f"{best_score:.4f}")
                st.progress(min(best_score, 1.0))

                img_path = os.path.join(engine.project_root, "captured_faces", f"{best_label}.jpg")
                if os.path.exists(img_path):
                    st.image(img_path, caption=f"Matched Image: {best_label}", use_container_width=True)

                if len(results) > 1:
                    st.write("### Other Close Matches:")
                    for lbl, sim in results[1:]:
                        st.write(f"- {lbl} (similarity: {sim:.4f})")
            else:
                st.warning("🆕 No confident match found.")
                st.metric("Best Similarity", f"{best_score:.4f}")
                st.progress(min(best_score, 1.0))

        except Exception as e:
            st.error(f"Error during matching: {e}")

# -------------------------------------------------------------
# Footer
# -------------------------------------------------------------
st.info("💡 Tip: Use 'Register New Face' to enroll people, then test them in 'Match Existing Face'.")
