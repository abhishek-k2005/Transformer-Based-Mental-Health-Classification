"""
app.py
------
Streamlit inference application for the Transformer-Based Mental Health
Classification project.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from config import APP_SUBTITLE, APP_TITLE, LABEL_COLORS
from inference import load_model_and_tokenizer, predict

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Global styles ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Background ── */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        min-height: 100vh;
    }

    /* ── Hero header ── */
    .hero {
        text-align: center;
        padding: 2.5rem 1rem 1.5rem;
    }
    .hero h1 {
        font-size: 2.6rem;
        font-weight: 700;
        background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.4rem;
    }
    .hero p {
        color: #94a3b8;
        font-size: 1rem;
        font-weight: 400;
        margin: 0;
    }

    /* ── Input card ── */
    .input-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.10);
        border-radius: 16px;
        padding: 1.6rem;
        backdrop-filter: blur(10px);
        margin-bottom: 1.2rem;
    }

    /* ── Result card ── */
    .result-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.10);
        border-radius: 16px;
        padding: 1.8rem;
        backdrop-filter: blur(10px);
        margin-top: 1rem;
        animation: fadeIn 0.5s ease;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to   { opacity: 1; transform: translateY(0);    }
    }

    /* ── Prediction badge ── */
    .pred-badge {
        display: inline-block;
        padding: 0.45rem 1.1rem;
        border-radius: 999px;
        font-size: 1rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        margin-bottom: 0.6rem;
    }

    /* ── Confidence text ── */
    .confidence-text {
        font-size: 0.95rem;
        color: #94a3b8;
        margin-bottom: 1.4rem;
    }

    /* ── Probability bars ── */
    .prob-row {
        display: flex;
        align-items: center;
        margin-bottom: 0.55rem;
        gap: 0.5rem;
    }
    .prob-label {
        width: 160px;
        min-width: 160px;
        font-size: 0.85rem;
        color: #cbd5e1;
        font-weight: 500;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .prob-bar-bg {
        flex: 1;
        background: rgba(255,255,255,0.07);
        border-radius: 999px;
        height: 10px;
        overflow: hidden;
    }
    .prob-bar-fill {
        height: 100%;
        border-radius: 999px;
        transition: width 0.6s ease;
    }
    .prob-pct {
        width: 44px;
        text-align: right;
        font-size: 0.82rem;
        color: #94a3b8;
        font-variant-numeric: tabular-nums;
    }

    /* ── Disclaimer ── */
    .disclaimer {
        text-align: center;
        font-size: 0.78rem;
        color: #64748b;
        margin-top: 2rem;
        padding-bottom: 1rem;
    }

    /* ── Streamlit element overrides ── */
    div[data-testid="stTextArea"] textarea {
        background: rgba(255,255,255,0.06) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 10px !important;
        color: #e2e8f0 !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.95rem !important;
        resize: vertical !important;
    }
    div[data-testid="stTextArea"] textarea:focus {
        border-color: #7c3aed !important;
        box-shadow: 0 0 0 2px rgba(124,58,237,0.3) !important;
    }
    div[data-testid="stTextArea"] label {
        color: #cbd5e1 !important;
        font-weight: 500 !important;
    }

    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.7rem 1rem !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.02em !important;
        transition: opacity 0.2s, transform 0.15s !important;
        font-family: 'Inter', sans-serif !important;
    }
    .stButton > button:hover {
        opacity: 0.88 !important;
        transform: translateY(-1px) !important;
    }
    .stButton > button:active {
        transform: translateY(0) !important;
    }

    /* ── Warning / info tweaks ── */
    div[data-testid="stAlert"] {
        border-radius: 10px;
        border: none;
    }

    /* ── Section divider ── */
    .section-heading {
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #64748b;
        margin-bottom: 0.9rem;
    }

    /* ── Spinner overlay colour ── */
    .stSpinner > div { border-top-color: #7c3aed !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Model loading (cached) ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model — this may take a moment…")
def get_model():
    """Load tokeniser + model once and cache for the session lifetime."""
    return load_model_and_tokenizer()


# ── Helper: build the probability bars HTML ───────────────────────────────────
def _build_prob_bars(probabilities: dict[str, float], highlight: str) -> str:
    default_color = "#60a5fa"
    rows = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    html_parts = []
    for label, prob in rows:
        pct = prob * 100
        color = LABEL_COLORS.get(label, default_color)
        is_top = label == highlight
        label_style = "font-weight:700; color:#e2e8f0;" if is_top else ""
        html_parts.append(
            f"""
            <div class="prob-row">
              <div class="prob-label" style="{label_style}">{label}</div>
              <div class="prob-bar-bg">
                <div class="prob-bar-fill"
                     style="width:{pct:.1f}%; background:{color};
                            {'box-shadow:0 0 6px ' + color + '88;' if is_top else ''}">
                </div>
              </div>
              <div class="prob-pct">{pct:.1f}%</div>
            </div>
            """
        )
    return "".join(html_parts)


# ── Hero section ──────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div class="hero">
      <h1>🧠 {APP_TITLE}</h1>
      <p>{APP_SUBTITLE}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Load model ────────────────────────────────────────────────────────────────
try:
    tokenizer, model, id2label = get_model()
    model_loaded = True
except Exception as exc:
    model_loaded = False
    st.error(
        f"**Could not load the model.** Make sure `models/final_model/` exists and "
        f"contains the required files.\n\n`{exc}`"
    )

# ── Input area ────────────────────────────────────────────────────────────────
st.markdown('<div class="input-card">', unsafe_allow_html=True)

user_text = st.text_area(
    label="Enter text to classify",
    placeholder=(
        "Paste or type any text here — a journal entry, a social-media post, "
        "or a short paragraph — and the model will predict its mental-health category."
    ),
    height=180,
    key="user_input",
)

analyze_clicked = st.button("🔍 Analyze Text", disabled=not model_loaded, key="analyze_btn")

st.markdown("</div>", unsafe_allow_html=True)

# ── Inference & results ───────────────────────────────────────────────────────
if analyze_clicked:
    text_stripped = (user_text or "").strip()

    if not text_stripped:
        st.warning("⚠️  Please enter some text before clicking **Analyze Text**.")
    else:
        with st.spinner("Analyzing…"):
            try:
                result = predict(text_stripped, tokenizer, model, id2label)
            except Exception as exc:
                st.error(f"Inference failed: {exc}")
                st.stop()

        pred_label = result["predicted_label"]
        confidence = result["confidence"]
        probabilities = result["probabilities"]

        badge_color = LABEL_COLORS.get(pred_label, "#60a5fa")

        # Result card
        prob_bars_html = _build_prob_bars(probabilities, pred_label)
        st.markdown(
            f"""
            <div class="result-card">
              <div class="section-heading">Prediction</div>
              <div>
                <span class="pred-badge"
                      style="background:{badge_color}22; color:{badge_color};
                             border:1px solid {badge_color}55;">
                  {pred_label}
                </span>
              </div>
              <div class="confidence-text">
                Confidence: <strong style="color:#e2e8f0;">{confidence*100:.1f}%</strong>
              </div>

              <div class="section-heading">Class probabilities</div>
              {prob_bars_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

# ── Footer disclaimer ─────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="disclaimer">
      ⚠️ This tool is for research and portfolio purposes only.<br>
      It is <strong>not</strong> a clinical diagnostic instrument.<br>
      If you or someone you know is struggling, please seek professional support.
    </div>
    """,
    unsafe_allow_html=True,
)
