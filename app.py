# =============================================================================
# Vehicle Damage Classification for Insurance Claim Assessment
# University Thesis — Proof of Concept Streamlit Application
#
# Required packages:
#   pip install streamlit tensorflow pillow numpy pandas matplotlib
#
# Usage:
#   streamlit run app.py
#
# Expected model files:
#   models/mobilenetv2_finetuned_best.keras
#   models/densenet121_finetuned_best.keras
# =============================================================================

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must be set before pyplot import
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------------
# TensorFlow — deferred import so startup is fast and errors surface cleanly
# ---------------------------------------------------------------------------
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.applications.mobilenet_v2 import (
        preprocess_input as mobilenetv2_preprocess,
    )
    from tensorflow.keras.applications.densenet import (
        preprocess_input as densenet_preprocess,
    )
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


# =============================================================================
# CSS LOADER
# =============================================================================

def load_css(css_file: str) -> None:
    """
    Read an external CSS file and inject it into the Streamlit page.

    All application styles live in style.css; this keeps app.py focused
    purely on logic and layout rather than visual presentation rules.

    Parameters
    ----------
    css_file : str
        Path to the .css file, relative to the working directory.

    Raises
    ------
    FileNotFoundError
        Surfaced as a Streamlit warning so the app still renders.
    """
    try:
        css_content = Path(css_file).read_text(encoding="utf-8")
        st.markdown(f"<style>\n{css_content}\n</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(
            f"⚠️ Stylesheet not found: `{css_file}`. "
            "The app will render with default styles.",
            icon="🎨",
        )


# =============================================================================
# CONSTANTS
# =============================================================================

APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "models"
STYLE_FILE = APP_DIR / "style.css"

# Class labels — must mirror the directory sort order used during training:
#   index 0 → label '1',  index 1 → label '2', …,  index 4 → label '5'
CLASS_NAMES: list[str] = ["1", "2", "3", "4", "5"]

# Spatial dimensions expected by both fine-tuned models
IMAGE_SIZE: tuple[int, int] = (190, 190)

# Model identifiers used as dict keys throughout the app
MODEL_MOBILENETV2 = "MobileNetV2"
MODEL_DENSENET121  = "DenseNet121"

# Per-model static configuration (populated after TF import guard below)
MODEL_CONFIGS: dict[str, dict] = {
    MODEL_MOBILENETV2: {
        "path": MODELS_DIR / "mobilenetv2_finetuned_best.keras",
        "preprocess": mobilenetv2_preprocess if TF_AVAILABLE else None,
        "description": (
            "Lightweight inverted-residual CNN optimised for mobile & "
            "edge devices. Excellent accuracy-to-latency trade-off."
        ),
        "params": "3.4 M",
        "icon": "⚡",
    },
    MODEL_DENSENET121: {
        "path": MODELS_DIR / "densenet121_finetuned_best.keras",
        "preprocess": densenet_preprocess if TF_AVAILABLE else None,
        "description": (
            "121-layer densely connected CNN. Each layer receives feature "
            "maps from all preceding layers, reducing vanishing-gradient issues."
        ),
        "params": "8.1 M",
        "icon": "🧠",
    },
}

# ── Severity scale ──────────────────────────────────────────────────────────
SEVERITY_INTERPRETATION: dict[str, dict] = {
    "1": {
        "label":       "Minor / Cosmetic",
        "full_label":  "Minor / Cosmetic Damage",
        "colour":      "#00c896",   # teal-green
        "badge":       "🟢",
        "cost_range":  "< $500",
        "description": (
            "Superficial scratches, paint chips, or very small scuffs. "
            "Vehicle functionality is unaffected. "
            "Typically resolved through a paint touch-up or minor panel work."
        ),
    },
    "2": {
        "label":       "Low Severity",
        "full_label":  "Low Severity Damage",
        "colour":      "#f5c518",   # amber
        "badge":       "🟡",
        "cost_range":  "$500 – $2,000",
        "description": (
            "Small dents or localised panel deformation. "
            "Vehicle remains fully roadworthy. "
            "Moderate repair costs; standard body-shop intervention."
        ),
    },
    "3": {
        "label":       "Moderate Severity",
        "full_label":  "Moderate Severity Damage",
        "colour":      "#ff8c42",   # orange
        "badge":       "🟠",
        "cost_range":  "$2,000 – $6,000",
        "description": (
            "Visible structural deformation or multiple panels affected. "
            "Professional assessment required before driving. "
            "Significant repair costs anticipated."
        ),
    },
    "4": {
        "label":       "High Severity",
        "full_label":  "High Severity Damage",
        "colour":      "#e05252",   # red
        "badge":       "🔴",
        "cost_range":  "$6,000 – $15,000",
        "description": (
            "Major structural damage involving safety-critical components. "
            "Vehicle likely undriveable or unsafe. "
            "High repair bill or partial write-off possible."
        ),
    },
    "5": {
        "label":       "Total Loss",
        "full_label":  "Very High Severity / Total Loss",
        "colour":      "#a855f7",   # violet
        "badge":       "🟣",
        "cost_range":  "> $15,000",
        "description": (
            "Extensive damage across multiple critical systems. "
            "Vehicle is almost certainly a total economic loss. "
            "Immediate specialist inspection mandatory; write-off likely."
        ),
    },
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_model_config(model_name: str) -> dict:
    """
    Return the static configuration dict for a given model name.

    Parameters
    ----------
    model_name : str  — one of MODEL_MOBILENETV2 or MODEL_DENSENET121

    Returns
    -------
    dict with keys: path, preprocess, description, params, icon

    Raises
    ------
    ValueError if model_name is not recognised.
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Valid options: {list(MODEL_CONFIGS.keys())}"
        )
    return MODEL_CONFIGS[model_name]


def clear_prediction_state() -> None:
    """Remove cached prediction data when the current inputs change."""
    for key in ("result", "model_used", "result_signature"):
        st.session_state.pop(key, None)


def build_result_signature(
    model_name: str,
    uploaded_file,
) -> tuple[str, str | None, int | None]:
    """Create a small signature for the active model and uploaded image."""
    if uploaded_file is None:
        return model_name, None, None
    return model_name, uploaded_file.name, uploaded_file.size


@st.cache_resource(show_spinner=False)
def load_selected_model(model_path: str):
    """
    Load and cache a Keras model from disk.

    Streamlit's @st.cache_resource decorator ensures the model is loaded
    only once per unique path across all reruns — switching models reloads
    only the newly selected one.

    Parameters
    ----------
    model_path : str — absolute or relative path to the .keras file

    Returns
    -------
    tf.keras.Model

    Raises
    ------
    FileNotFoundError if the file is absent.
    """
    if not Path(model_path).is_file():
        raise FileNotFoundError(
            f"Model file not found: '{model_path}'\n"
            "Place the trained .keras file in the models/ directory."
        )
    return load_model(model_path, compile=False)


def preprocess_uploaded_image(image: Image.Image, model_name: str) -> np.ndarray:
    """
    Convert a raw PIL image into a model-ready float32 batch tensor.

    Pipeline
    --------
    1. Convert to RGB  (handles RGBA / greyscale / palette modes)
    2. Resize to IMAGE_SIZE using Lanczos resampling
    3. Cast to float32 NumPy array
    4. Add batch dimension → shape (1, 190, 190, 3)
    5. Apply model-specific normalisation

    Parameters
    ----------
    image      : PIL.Image.Image
    model_name : str — selects the correct preprocessing function

    Returns
    -------
    np.ndarray of shape (1, 190, 190, 3)
    """
    config       = get_model_config(model_name)
    preprocess_fn = config["preprocess"]

    img_rgb      = image.convert("RGB")
    img_resized  = img_rgb.resize(IMAGE_SIZE, Image.LANCZOS)
    img_array    = np.array(img_resized, dtype=np.float32)
    img_batch    = np.expand_dims(img_array, axis=0)          # (1, 190, 190, 3)
    return preprocess_fn(img_batch)


def predict_damage(
    model,
    processed_image: np.ndarray,
    class_names: list[str],
) -> dict:
    """
    Run forward pass and return a structured result dictionary.

    Parameters
    ----------
    model           : tf.keras.Model
    processed_image : np.ndarray — shape (1, 190, 190, 3)
    class_names     : list[str]  — ordered label list

    Returns
    -------
    dict
        predicted_class : str   — e.g. '3'
        confidence      : float — top-1 probability ∈ [0, 1]
        probabilities   : dict  — {label: probability}
        class_index     : int   — argmax index
    """
    raw_preds   = model.predict(processed_image, verbose=0)   # (1, N)
    probs       = raw_preds[0]                                 # (N,)
    class_index = int(np.argmax(probs))
    confidence  = float(probs[class_index])
    pred_class  = class_names[class_index]

    return {
        "predicted_class": pred_class,
        "confidence":      confidence,
        "probabilities":   {lbl: float(p) for lbl, p in zip(class_names, probs)},
        "class_index":     class_index,
    }


def build_probability_chart(probabilities: dict, predicted_class: str) -> plt.Figure:
    """
    Render a polished horizontal bar chart of per-class probabilities.

    The predicted class bar is rendered in its severity colour;
    all others use a muted neutral tone.

    Parameters
    ----------
    probabilities   : dict  — {class_label: float probability}
    predicted_class : str   — highlighted class label

    Returns
    -------
    matplotlib.figure.Figure
    """
    BG      = "#0d1117"
    NEUTRAL = "#2d3748"

    labels  = list(probabilities.keys())
    values  = [probabilities[lbl] * 100 for lbl in labels]
    colours = [
        SEVERITY_INTERPRETATION[lbl]["colour"] if lbl == predicted_class else NEUTRAL
        for lbl in labels
    ]

    fig, ax = plt.subplots(figsize=(7, 3.8))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    bars = ax.barh(
        [f"Class {lbl}" for lbl in labels],
        values,
        color=colours,
        height=0.52,
        edgecolor="none",
    )

    for i, (bar, val) in enumerate(zip(bars, values)):
        lbl      = labels[i]
        is_pred  = lbl == predicted_class
        x_offset = min(val + 1.2, 97)
        ax.text(
            x_offset,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%",
            va="center",
            ha="left",
            fontsize=10,
            color=SEVERITY_INTERPRETATION[lbl]["colour"] if is_pred else "#8899aa",
            fontweight="bold" if is_pred else "normal",
        )

    ax.set_xlim(0, 110)
    ax.set_xlabel("Confidence (%)", color="#8899aa", fontsize=9, labelpad=8)
    ax.set_title(
        "Prediction Confidence by Class",
        color="#e2e8f0",
        fontsize=11,
        fontweight="semibold",
        pad=12,
    )
    ax.tick_params(colors="#8899aa", labelsize=9)
    ax.xaxis.set_tick_params(length=0)
    ax.yaxis.set_tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.axvline(x=50, color="#2d3748", linewidth=0.7, linestyle="--")

    ax.invert_yaxis()
    plt.tight_layout(pad=1.2)
    return fig


# =============================================================================
# PAGE CONFIG  — must be the first Streamlit call
# =============================================================================

st.set_page_config(
    page_title="VehicleDX — Insurance Claim Classifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "VehicleDX · Vehicle Damage Classification for Motor Insurance "
            "Claim Assessment · University Thesis Proof-of-Concept."
        )
    },
)


# =============================================================================
# STYLES — viewport + Google Font + external stylesheet
# =============================================================================

# Viewport meta tag — CRITICAL for mobile: without this, mobile browsers
# render at desktop width and scale down, ignoring all media queries.
# Google Fonts must also be a <link> tag because Streamlit's sandbox
# blocks CDN @import rules inside a locally served .css file.
st.markdown(
    """
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=5.0">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap"
          rel="stylesheet">
    """,
    unsafe_allow_html=True,
)

# All visual rules are maintained in style.css — edit that file to change the look.
load_css(str(STYLE_FILE))


with st.sidebar:

    # ── Brand mark ────────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="brand-icon">🛡️</div>
            <div>
                <div class="brand-name">VehicleDX</div>
                <div class="brand-tagline">Damage Classification</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Model selection ───────────────────────────────────────────────────
    st.markdown('<p class="section-label">Active Model</p>', unsafe_allow_html=True)

    selected_model_name = st.selectbox(
        label="model_selector",
        options=[MODEL_MOBILENETV2, MODEL_DENSENET121],
        index=0,
        label_visibility="collapsed",
        help="Choose which fine-tuned CNN backbone to use for inference.",
    )

    cfg = get_model_config(selected_model_name)

    st.markdown(
        f"""
        <div class="model-card">
            <div class="mc-name">{cfg['icon']} {selected_model_name}</div>
            <div class="mc-desc">{cfg['description']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── How-to steps ──────────────────────────────────────────────────────
    st.markdown('<p class="section-label">How to Use</p>', unsafe_allow_html=True)
    steps = [
        ("1", "Select a <strong>model</strong> from the dropdown above."),
        ("2", "Upload a <strong>JPG / JPEG / PNG</strong> photo of the damaged vehicle."),
        ("3", "Press <strong>Classify Damage</strong> to run inference."),
        ("4", "Review the <strong>severity class</strong> and probability breakdown."),
    ]
    for num, text in steps:
        st.markdown(
            f"""
            <div class="step-item">
                <div class="step-num">{num}</div>
                <div class="step-text">{text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Academic disclaimer ────────────────────────────────────────────────
    st.markdown(
        """
        <div class="academic-badge">
            <div class="ab-title">🎓 Academic Proof of Concept</div>
            <div class="ab-text">
                Built for university thesis purposes only.
                Predictions are <strong>not legally binding</strong> and do
                not replace a professional vehicle inspection or adjuster review.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# HERO HEADER
# =============================================================================

st.markdown(
    """
    <div class="hero-wrapper">
        <div class="hero-badge">🛡️ &nbsp; Motor Insurance · Deep Learning · PoC</div>
        <h1 class="hero-title">Vehicle Damage Classification</h1>
        <p class="hero-subtitle">Automated Severity Assessment for Insurance Claim Triage</p>
        <p class="hero-description">
            A proof-of-concept system powered by fine-tuned deep learning models trained
            to classify vehicle damage severity across five levels — from minor cosmetic
            damage to total economic loss — supporting faster preliminary triage of motor
            insurance claims.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# TensorFlow guard
# =============================================================================

if not TF_AVAILABLE:
    st.error(
        "❌ **TensorFlow is not installed.** "
        "Run: `pip install tensorflow` and restart the app.",
        icon="🔴",
    )
    st.stop()


# =============================================================================
# INLINE STAT CARDS
# =============================================================================
# Rendered as a single HTML CSS-grid div (class="stat-cards-row") so that the
# responsive breakpoints in style.css can freely reflow the grid from 4→2→1
# columns. Streamlit st.columns() is a flex row that cannot be overridden by
# CSS media queries reliably across all browsers.

stat_items = [
    (cfg["icon"] + "\u00a0" + selected_model_name, "Active Model"),
    (f"{IMAGE_SIZE[0]}\u00a0×\u00a0{IMAGE_SIZE[1]}", "Input Resolution (px)"),
    ("5", "Damage Severity Classes"),
    (cfg["params"], "Model Parameters"),
]

stat_cards_html = '<div class="stat-cards-row">'
for value, label in stat_items:
    stat_cards_html += f"""
        <div class="stat-card">
            <div class="stat-value">{value}</div>
            <div class="stat-label">{label}</div>
        </div>"""
stat_cards_html += "\n</div>"

st.markdown(stat_cards_html, unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# =============================================================================
# IMAGE UPLOAD
# =============================================================================

st.markdown(
    '<p class="section-label">① &nbsp; Upload Vehicle Image</p>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    label="Drag & drop or click to browse",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, PNG. Use clear, well-lit photographs.",
)

current_signature = build_result_signature(selected_model_name, uploaded_file)
if st.session_state.get("result_signature") != current_signature:
    clear_prediction_state()


# =============================================================================
# INFERENCE PIPELINE
# =============================================================================

if uploaded_file is not None:

    # ── Parse image ─────────────────────────────────────────────────────────
    try:
        raw_image = Image.open(uploaded_file)
    except Exception as exc:
        st.error(
            f"❌ **Invalid image.** Could not open the uploaded file.\n\nDetails: `{exc}`",
            icon="🔴",
        )
        st.stop()

    # ── Two-column layout: image | controls ─────────────────────────────────
    col_img, col_ctrl = st.columns([1.6, 1], gap="large")

    with col_img:
        st.markdown(
            '<p class="section-label">② &nbsp; Uploaded Photo</p>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="img-panel">', unsafe_allow_html=True)
        st.image(raw_image, width="stretch")
        st.markdown(
            f"""
            <div class="img-meta">
                {raw_image.width}&thinsp;×&thinsp;{raw_image.height} px
                &nbsp;·&nbsp; Mode: {raw_image.mode}
                &nbsp;·&nbsp; → resized to {IMAGE_SIZE[0]}&thinsp;×&thinsp;{IMAGE_SIZE[1]} px for inference
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col_ctrl:
        st.markdown(
            '<p class="section-label">③ &nbsp; Run Classification</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div style="background:var(--bg-glass); border:1px solid var(--border);
                        border-radius:var(--radius-md); padding:1rem 1.1rem; margin-bottom:1rem;">
                <div style="font-size:0.78rem; color:var(--text-muted); margin-bottom:0.5rem;
                             text-transform:uppercase; letter-spacing:0.07em; font-weight:600;">
                    Selected backbone
                </div>
                <div style="font-size:1rem; font-weight:700; color:var(--text-primary);">
                    {cfg['icon']} &nbsp;{selected_model_name}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        run_prediction = st.button(
            "🔍 &nbsp; Classify Damage",
            type="primary",
            width="stretch",
        )

        if run_prediction:
            # ── Load model ────────────────────────────────────────────────
            with st.spinner(f"Loading {selected_model_name}…"):
                try:
                    model = load_selected_model(cfg["path"])
                except FileNotFoundError as exc:
                    st.error(str(exc), icon="📁")
                    st.stop()
                except Exception as exc:
                    st.error(f"❌ Model load failed: `{exc}`", icon="🔴")
                    st.stop()

            # ── Preprocess ────────────────────────────────────────────────
            with st.spinner("Pre-processing image…"):
                try:
                    processed = preprocess_uploaded_image(raw_image, selected_model_name)
                except Exception as exc:
                    st.error(f"❌ Pre-processing failed: `{exc}`", icon="🔴")
                    st.stop()

            # ── Predict ───────────────────────────────────────────────────
            with st.spinner("Running inference…"):
                try:
                    result = predict_damage(model, processed, CLASS_NAMES)
                except Exception as exc:
                    st.error(f"❌ Prediction failed: `{exc}`", icon="🔴")
                    st.stop()

            # Persist result across reruns
            st.session_state["result"] = result
            st.session_state["model_used"] = selected_model_name
            st.session_state["result_signature"] = current_signature

        if "result" not in st.session_state:
            st.markdown(
                """
                <div style="background:rgba(59,130,246,0.06); border:1px solid rgba(59,130,246,0.2);
                            border-radius:var(--radius-md); padding:1rem; margin-top:0.5rem;
                            font-size:0.82rem; color:#93c5fd; line-height:1.55;">
                    💡 Press <strong>Classify Damage</strong> above to run the selected
                    model on the uploaded image.
                </div>
                """,
                unsafe_allow_html=True,
            )

    # =========================================================================
    # RESULTS  (displayed once a prediction is available)
    # =========================================================================

    if "result" in st.session_state:
        result     = st.session_state["result"]
        model_used = st.session_state["model_used"]

        pred_class = result["predicted_class"]
        confidence = result["confidence"]
        probs      = result["probabilities"]
        severity   = SEVERITY_INTERPRETATION[pred_class]
        clr        = severity["colour"]

        st.markdown("<hr>", unsafe_allow_html=True)

        # ── Section heading ────────────────────────────────────────────────
        st.markdown(
            '<p class="section-label">④ &nbsp; Classification Results</p>',
            unsafe_allow_html=True,
        )

        # ── Result hero card ───────────────────────────────────────────────
        st.markdown(
            f"""
            <div class="result-hero" style="
                background: linear-gradient(135deg,
                    {clr}22 0%,
                    {clr}10 60%,
                    transparent 100%);
                border: 1px solid {clr}44;
            ">
                <div class="result-hero-inner">
                    <span class="result-class-badge"
                          style="background:{clr}22; border:1px solid {clr}66; color:{clr};">
                        {severity['badge']} &nbsp; Class {pred_class}
                    </span>
                    <div class="result-class-name" style="color:{clr};">
                        {severity['full_label']}
                    </div>
                    <p class="result-description">{severity['description']}</p>
                    <div style="margin-top:1rem; display:flex; gap:2rem; flex-wrap:wrap;">
                        <div>
                            <div style="font-size:0.68rem; color:rgba(226,232,240,0.45);
                                        text-transform:uppercase; letter-spacing:0.08em;
                                        font-weight:600; margin-bottom:2px;">Confidence</div>
                            <div style="font-size:1.5rem; font-weight:800; color:{clr};">
                                {confidence * 100:.1f}%
                            </div>
                        </div>
                        <div>
                            <div style="font-size:0.68rem; color:rgba(226,232,240,0.45);
                                        text-transform:uppercase; letter-spacing:0.08em;
                                        font-weight:600; margin-bottom:2px;">Est. Repair Cost</div>
                            <div style="font-size:1.5rem; font-weight:800; color:var(--text-primary);">
                                {severity['cost_range']}
                            </div>
                        </div>
                        <div>
                            <div style="font-size:0.68rem; color:rgba(226,232,240,0.45);
                                        text-transform:uppercase; letter-spacing:0.08em;
                                        font-weight:600; margin-bottom:2px;">Model</div>
                            <div style="font-size:1.5rem; font-weight:800; color:var(--text-primary);">
                                {model_used}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Chart  +  inline confidence bars ───────────────────────────────
        chart_col, bars_col = st.columns([1.55, 1], gap="large")

        with chart_col:
            st.markdown(
                '<p class="section-label">Probability Distribution</p>',
                unsafe_allow_html=True,
            )
            fig = build_probability_chart(probs, pred_class)
            st.pyplot(fig, width="stretch")
            plt.close(fig)

        with bars_col:
            st.markdown(
                '<p class="section-label">Confidence Breakdown</p>',
                unsafe_allow_html=True,
            )

            # Render animated CSS confidence bars
            bars_html = ""
            for lbl, prob in probs.items():
                is_pred   = lbl == pred_class
                bar_color = SEVERITY_INTERPRETATION[lbl]["colour"] if is_pred else "#2d3748"
                pct       = prob * 100
                weight    = "700" if is_pred else "400"
                txt_col   = SEVERITY_INTERPRETATION[lbl]["colour"] if is_pred else "#8899aa"

                bars_html += f"""
                <div class="confidence-row">
                    <span class="conf-label" style="color:{txt_col}; font-weight:{weight};">
                        {SEVERITY_INTERPRETATION[lbl]['badge']} C{lbl}
                    </span>
                    <div class="conf-bar-bg">
                        <div class="conf-bar-fill"
                             style="width:{pct:.1f}%; background:{bar_color};"></div>
                    </div>
                    <span class="conf-pct" style="color:{txt_col}; font-weight:{weight};">
                        {pct:.1f}%
                    </span>
                </div>
                """

            st.markdown(bars_html, unsafe_allow_html=True)

            # Probability table
            st.markdown("<br>", unsafe_allow_html=True)
            df = pd.DataFrame({
                "Class":    [f"Class {lbl}" for lbl in probs],
                "Label":    [SEVERITY_INTERPRETATION[lbl]["label"] for lbl in probs],
                "Prob (%)": [f"{v * 100:.2f}" for v in probs.values()],
            })
            st.dataframe(df, hide_index=True, width="stretch")

        # ── Insurance interpretation panel ─────────────────────────────────
        # Rendered as a single HTML CSS-grid div (class="sev-cards-row") for
        # the same reason as stat_cards_row — full CSS breakpoint control.
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(
            '<p class="section-label">⑤ &nbsp; Insurance Severity Scale</p>',
            unsafe_allow_html=True,
        )

        sev_cards_html = '<div class="sev-cards-row">'
        for cls_lbl, info in SEVERITY_INTERPRETATION.items():
            is_pred = cls_lbl == pred_class
            border  = f"2px solid {info['colour']}" if is_pred else "1px solid rgba(255,255,255,0.07)"
            bg      = f"{info['colour']}18"         if is_pred else "rgba(255,255,255,0.03)"
            shadow  = f"0 4px 20px {info['colour']}33" if is_pred else "none"
            sev_cards_html += f"""
            <div class="sev-card"
                 style="border:{border}; background:{bg}; box-shadow:{shadow};">
                <div class="sev-badge">{info['badge']}</div>
                <div class="sev-class" style="color:{info['colour']};">
                    Class {cls_lbl}
                </div>
                <div class="sev-name">{info['label']}</div>
                <div class="sev-cost">{info['cost_range']}</div>
            </div>"""
        sev_cards_html += "\n</div>"
        st.markdown(sev_cards_html, unsafe_allow_html=True)

        # ── Legal disclaimer ───────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            """
            <div style="background:rgba(245,197,24,0.07); border:1px solid rgba(245,197,24,0.25);
                        border-radius:var(--radius-md); padding:1rem 1.2rem;
                        font-size:0.82rem; color:rgba(226,232,240,0.7); line-height:1.6;">
                ⚖️ &nbsp;<strong style="color:#f5c518;">Important Notice:</strong>
                This classification represents a <strong>preliminary automated assessment only</strong>.
                It is intended to <em>support</em> — not replace — qualified professional vehicle inspection
                and adjuster evaluation. Final claim decisions must follow established regulatory and
                underwriting procedures.
            </div>
            """,
            unsafe_allow_html=True,
        )

else:
    # =========================================================================
    # EMPTY STATE — no image uploaded yet
    # =========================================================================
    st.markdown(
        """
        <div class="empty-state">
            <div class="es-icon">🚘</div>
            <h3>No Image Uploaded Yet</h3>
            <p>
                Use the file uploader above to select a <strong>JPG, JPEG, or PNG</strong>
                photograph of a damaged vehicle. The system will automatically classify
                the damage severity across five levels and provide a probability breakdown.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# FOOTER
# =============================================================================

st.markdown(
    """
    <div class="app-footer">
        <strong>VehicleDX</strong>
        <span class="footer-divider">·</span>
        Vehicle Damage Classification for Motor Insurance Claim Assessment
        <span class="footer-divider">·</span>
        University Thesis Proof of Concept
        <br>
        Models: MobileNetV2 &amp; DenseNet121 (fine-tuned)
        <span class="footer-divider">·</span>
        Input: 190 × 190 px
        <span class="footer-divider">·</span>
        Built with
        <a href="https://streamlit.io" target="_blank">Streamlit</a>
        &amp;
        <a href="https://www.tensorflow.org" target="_blank">TensorFlow</a>
    </div>
    """,
    unsafe_allow_html=True,
)
