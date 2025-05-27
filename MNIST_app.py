import torch
from model import get_model
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from torchvision import transforms
from PIL import Image, ImageOps
import numpy as np

# Load model
Net = get_model()
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
model = Net()
model.load_state_dict(torch.load("model/mnist.pth", map_location=device))
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

# --- Sidebar ---
st.sidebar.title("MNIST Digit Classifier")
st.sidebar.markdown(f"**Device in use:** `{device}`")
st.sidebar.markdown("Draw a digit or upload an image (28x28 grayscale).")

# --- Title & Subtitle ---
st.title("Handwritten Digit Recognition")
st.markdown("Draw a digit or upload an image, then click **Predict** to classify it using a trained MNIST model.")

# Backgroud
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://i.imgur.com/1NTrY0M.png");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Tabs ---
tab1, tab2 = st.tabs(["Draw Digit", "Upload Image"])

def preprocess_image(img: Image.Image):
    img = img.convert("L")                       # Grayscale
    img = ImageOps.invert(img)                   # Invert for white digit on black
    tensor = transform(img)
    return tensor.unsqueeze(0), img

# --- Draw Digit Tab ---
with tab1:
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=12,
        stroke_color="#000000",
        background_color="#FFFFFF",
        height=480,
        width=705,
        drawing_mode="freedraw",
        key="canvas",
    )

    if st.button("Predict from Drawing"):
        if canvas_result.image_data is not None:
            img = Image.fromarray(np.uint8(canvas_result.image_data)).convert("L")
            input_batch, img_display = preprocess_image(img)

            with torch.no_grad():
                output = model(input_batch)
                prediction = output.argmax(dim=1).item()
                probabilities = torch.nn.functional.softmax(output, dim=1).squeeze().numpy()

            st.image(img_display.resize((140, 140)), caption="Your Drawing")
            st.metric("🎯 Predicted Digit", prediction)
            st.subheader("📊 Prediction Probabilities")
            st.bar_chart(probabilities)
        else:
            st.warning("⚠️ Please draw a digit first.")

# --- Upload Image Tab ---
with tab2:
    uploaded_file = st.file_uploader("Upload a digit image (ideally 28x28 or larger)", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        input_batch, img_display = preprocess_image(image)

        with torch.no_grad():
            output = model(input_batch)
            prediction = output.argmax(dim=1).item()
            probabilities = torch.nn.functional.softmax(output, dim=1).squeeze().numpy()

        st.image(img_display.resize((140, 140)), caption="Uploaded Image")
        st.metric("🎯 Predicted Digit", prediction)
        st.subheader("📊 Prediction Probabilities")
        st.bar_chart(probabilities)

# Footer
st.markdown(
        """
        <hr style="margin-top: 50px;"/>
        <div style="text-align: center; padding: 10px;">
            <small>
                © 2025 <a href="https://www.linkedin.com/in/himel-sarder/" target="_blank">Himel Sarder</a> • All Rights Reserved <br/>
            </small>
        </div>
        """,
        unsafe_allow_html=True
    )
