import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# App config
st.set_page_config(page_title="MNIST Digit Classifier", layout="centered")

# Load model
model = load_model("mnist_model.h5")

# Sidebar info
st.sidebar.title("🧠 About the App")
st.sidebar.markdown("""
This is a simple deep learning demo that lets you draw a digit (0–9) and get real-time predictions from a neural network trained on the MNIST dataset.

🖌️ Just draw in the black box, click **Predict**, and see how well the model performs!
""")

# Main title
st.markdown("<h1 style='text-align: center;'>🔢 MNIST Digit Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Draw a digit (0–9) below and click <b>Predict</b> to see the model's guess.</p>", unsafe_allow_html=True)

# Drawing canvas
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=12,
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas"
)

# Predict button
if st.button("🎯 Predict", use_container_width=True):
    if canvas_result.image_data is not None:
        # Preprocess input image
        img = Image.fromarray((canvas_result.image_data[:, :, 0]).astype(np.uint8))
        img = img.resize((28, 28)).convert("L")
        img_array = np.array(img).reshape(1, 28, 28, 1).astype(np.float32) / 255.0

        # Predict
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)

        # Display result
        st.markdown(f"<h2 style='text-align: center;'>🧾 Predicted Digit: <span style='color: #4CAF50;'>{predicted_class}</span></h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center;'>Confidence: <b>{confidence:.2%}</b></p>", unsafe_allow_html=True)
    else:
        st.warning("Please draw something before predicting.")
