import os
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageFilter

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Neural Vision",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# MNIST IMAGE PREPROCESSING
# ============================================================

def preprocess_mnist_image(image):
    """
    Convert a handwritten digit image into MNIST-style input.

    Steps:
    1. Convert to grayscale
    2. Detect background
    3. Invert when necessary
    4. Detect/crop digit
    5. Add padding
    6. Center digit
    7. Resize to 28x28
    8. Normalize pixels to 0-1
    9. Add batch dimension
    """

    # --------------------------------------------------------
    # 1. Grayscale
    # --------------------------------------------------------

    image = image.convert("L")

    # Small blur to reduce camera/photo noise
    image = image.filter(
        ImageFilter.GaussianBlur(radius=0.5)
    )

    gray = np.array(image).astype(np.uint8)

    # --------------------------------------------------------
    # 2. Estimate background from image borders
    # --------------------------------------------------------

    border_pixels = np.concatenate(
        [
            gray[0, :],
            gray[-1, :],
            gray[:, 0],
            gray[:, -1],
        ]
    )

    background_value = np.median(border_pixels)

    # MNIST style:
    # black background = 0
    # white digit = 255

    if background_value > 127:
        gray = 255 - gray

    # --------------------------------------------------------
    # 3. Threshold
    # --------------------------------------------------------

    threshold = 50

    binary = gray > threshold

    # --------------------------------------------------------
    # 4. Find digit coordinates
    # --------------------------------------------------------

    coordinates = np.argwhere(binary)

    if coordinates.size == 0:
        return None, None, None

    y_min, x_min = coordinates.min(axis=0)
    y_max, x_max = coordinates.max(axis=0)

    # --------------------------------------------------------
    # 5. Crop digit
    # --------------------------------------------------------

    cropped = gray[
        y_min:y_max + 1,
        x_min:x_max + 1
    ]

    # --------------------------------------------------------
    # 6. Make square canvas + padding
    # --------------------------------------------------------

    h, w = cropped.shape

    padding = int(max(h, w) * 0.25)

    canvas_size = max(h, w) + (padding * 2)

    canvas = np.zeros(
        (canvas_size, canvas_size),
        dtype=np.uint8
    )

    # --------------------------------------------------------
    # 7. Center digit
    # --------------------------------------------------------

    y_offset = (canvas_size - h) // 2
    x_offset = (canvas_size - w) // 2

    canvas[
        y_offset:y_offset + h,
        x_offset:x_offset + w
    ] = cropped

    # --------------------------------------------------------
    # 8. Resize to MNIST 28x28
    # --------------------------------------------------------

    processed_image = Image.fromarray(canvas)

    processed_image = processed_image.resize(
        (28, 28),
        Image.Resampling.LANCZOS
    )

    # --------------------------------------------------------
    # 9. Normalize
    # --------------------------------------------------------

    normalized = (
        np.array(processed_image)
        .astype(np.float32)
        / 255.0
    )

    # --------------------------------------------------------
    # 10. Add batch dimension
    # --------------------------------------------------------

    model_input = normalized.reshape(
        1,
        28,
        28
    )

    return (
        processed_image,
        normalized,
        model_input
    )


# ============================================================
# SIMPLE THEME
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #071010;
    }

    [data-testid="stSidebar"] {
        background-color: #0b1717;
    }

    div[data-testid="stMetric"] {
        background-color: #0d1c1c;
        border: 1px solid #163636;
        border-radius: 12px;
        padding: 12px;
    }

    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# MODEL PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "mnist_model.keras"
)

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        return None

    return tf.keras.models.load_model(
        MODEL_PATH
    )


model = load_model()

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🧠 Neural Vision")

    st.caption(
        "Deep Learning • Computer Vision"
    )

    st.divider()

    st.subheader("About")

    st.write(
        "Neural Vision uses a trained neural network "
        "to recognize handwritten digits from 0 to 9."
    )

    st.divider()

    st.subheader("How it works")

    st.write("1. Upload a handwritten digit.")
    st.write("2. Convert image to grayscale.")
    st.write("3. Detect the digit.")
    st.write("4. Crop unnecessary background.")
    st.write("5. Center the digit.")
    st.write("6. Resize to 28 × 28.")
    st.write("7. Normalize pixels.")
    st.write("8. Predict using the neural network.")

    st.divider()

    st.subheader("Architecture")

    st.code(
        """Input
  ↓
Flatten
  ↓
Dense 128
  ↓
Dense 64
  ↓
Dense 10
  ↓
Softmax"""
    )

    if model is not None:
        st.success("Model loaded successfully")
    else:
        st.error("Model not found")

# ============================================================
# HEADER
# ============================================================

st.title("🧠 Neural Vision")

st.caption(
    "Handwritten Digit Recognition powered by a Neural Network"
)

st.write(
    "Upload a handwritten digit and let the neural network "
    "analyze it."
)

st.divider()

# ============================================================
# MODEL CHECK
# ============================================================

if model is None:

    st.error(
        "mnist_model.keras was not found."
    )

    st.info(
        "Make sure this file exists in the same folder:\n\n"
        "mnist_model.keras"
    )

    st.stop()

# ============================================================
# UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a handwritten digit image",
    type=[
        "png",
        "jpg",
        "jpeg"
    ],
    help="Upload an image containing one handwritten digit."
)

# ============================================================
# NO IMAGE
# ============================================================

if uploaded_file is None:

    st.info(
        "👆 Upload a handwritten digit image to begin."
    )

    st.subheader(
        "Neural Network Information"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Classes",
            "10"
        )
        st.caption(
            "Digits 0 → 9"
        )

    with col2:
        st.metric(
            "Input",
            "28 × 28"
        )
        st.caption(
            "MNIST format"
        )

    with col3:
        st.metric(
            "Output",
            "Softmax"
        )
        st.caption(
            "10 probabilities"
        )

    st.divider()

    st.subheader(
        "Model Architecture"
    )

    architecture = pd.DataFrame(
        {
            "Layer": [
                "Input",
                "Flatten",
                "Dense",
                "Dense",
                "Output"
            ],
            "Configuration": [
                "28 × 28",
                "784",
                "128 • ReLU",
                "64 • ReLU",
                "10 • Softmax"
            ]
        }
    )

    st.dataframe(
        architecture,
        use_container_width=True,
        hide_index=True
    )

    st.stop()

# ============================================================
# OPEN IMAGE
# ============================================================

try:

    original_image = Image.open(
        uploaded_file
    ).convert("RGB")

except Exception as e:

    st.error(
        f"Could not read image: {e}"
    )

    st.stop()

# ============================================================
# PREPROCESS IMAGE
# ============================================================

processed_image, normalized_image, model_input = (
    preprocess_mnist_image(
        original_image
    )
)

# ============================================================
# CHECK DIGIT DETECTION
# ============================================================

if processed_image is None:

    st.error(
        "No handwritten digit could be detected."
    )

    st.info(
        "Try uploading a clearer image containing "
        "one large handwritten digit."
    )

    st.stop()

# ============================================================
# IMAGE PREVIEW
# ============================================================

st.subheader(
    "🖼️ Image Processing"
)

col1, col2, col3 = st.columns(3)

with col1:

    st.write("Original")

    st.image(
        original_image,
        width="stretch"
    )

with col2:

    st.write("Cropped + Centered")

    st.image(
        processed_image,
        width="stretch"
    )

with col3:

    st.write("28 × 28 Normalized")

    st.image(
        normalized_image,
        width="stretch"
    )

# ============================================================
# PREDICTION
# ============================================================

st.divider()

st.subheader(
    "🔮 Prediction"
)

with st.spinner(
    "Neural network is analyzing the digit..."
):

    predictions = model.predict(
        model_input,
        verbose=0
    )

# ============================================================
# PROBABILITIES
# ============================================================

probabilities = predictions[0]

predicted_digit = int(
    np.argmax(probabilities)
)

confidence = (
    float(
        probabilities[predicted_digit]
    ) * 100
)

# ============================================================
# RESULT
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Predicted Digit",
        str(predicted_digit)
    )

with col2:

    st.metric(
        "Confidence",
        f"{confidence:.2f}%"
    )

with col3:

    st.metric(
        "Classes",
        "10"
    )

# ============================================================
# CONFIDENCE MESSAGE
# ============================================================

if confidence >= 90:

    st.success(
        f"Highly confident prediction: {predicted_digit}"
    )

elif confidence >= 70:

    st.warning(
        f"Predicted digit: {predicted_digit} "
        "with moderate confidence."
    )

else:

    st.warning(
        f"Predicted digit: {predicted_digit}, "
        "but confidence is relatively low."
    )

# ============================================================
# SOFTMAX PROBABILITIES
# ============================================================

st.divider()

st.subheader(
    "📊 Softmax Probabilities"
)

probability_df = pd.DataFrame(
    {
        "Digit": list(range(10)),
        "Probability": probabilities,
        "Percentage": [
            f"{float(p) * 100:.2f}%"
            for p in probabilities
        ]
    }
)

probability_df = probability_df.sort_values(
    "Probability",
    ascending=False
).reset_index(drop=True)

st.dataframe(
    probability_df,
    use_container_width=True,
    hide_index=True
)

# ============================================================
# BAR CHART
# ============================================================

st.subheader(
    "Prediction Distribution"
)

chart_df = pd.DataFrame(
    {
        "Digit": [
            str(i)
            for i in range(10)
        ],
        "Probability": probabilities
    }
)

chart_df = chart_df.set_index(
    "Digit"
)

st.bar_chart(
    chart_df,
    use_container_width=True
)

# ============================================================
# TOP 3
# ============================================================

st.divider()

st.subheader(
    "🏆 Top 3 Predictions"
)

top_indices = np.argsort(
    probabilities
)[::-1][:3]

c1, c2, c3 = st.columns(3)

columns = [
    c1,
    c2,
    c3
]

for position, (
    column,
    index
) in enumerate(
    zip(
        columns,
        top_indices
    ),
    start=1
):

    with column:

        st.metric(
            f"#{position}",
            f"Digit {int(index)}",
            f"{float(probabilities[index]) * 100:.2f}%"
        )

# ============================================================
# MODEL DETAILS
# ============================================================

st.divider()

st.subheader(
    "⚙️ Model Information"
)

c1, c2 = st.columns(2)

with c1:

    st.write("**Model file**")
    st.code("mnist_model.keras")

    st.write("**Input shape**")
    st.code("(1, 28, 28)")

    st.write("**Problem type**")
    st.code("Multiclass Classification")

with c2:

    st.write("**Hidden activation**")
    st.code("ReLU")

    st.write("**Output activation**")
    st.code("Softmax")

    st.write("**Number of classes**")
    st.code("10")

# ============================================================
# ARCHITECTURE TABLE
# ============================================================

st.subheader(
    "🧩 Neural Network Architecture"
)

architecture_df = pd.DataFrame(
    {
        "Layer": [
            "Input",
            "Flatten",
            "Dense",
            "Dense",
            "Output"
        ],
        "Units": [
            "28 × 28",
            "784",
            "128",
            "64",
            "10"
        ],
        "Activation": [
            "Pixels",
            "None",
            "ReLU",
            "ReLU",
            "Softmax"
        ],
        "Purpose": [
            "Receive image",
            "Flatten image",
            "Learn features",
            "Learn higher-level features",
            "Classify digit"
        ]
    }
)

st.dataframe(
    architecture_df,
    use_container_width=True,
    hide_index=True
)

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Neural Vision • MNIST Handwritten Digit Classification • "
    "Python + TensorFlow/Keras + Streamlit"
)

