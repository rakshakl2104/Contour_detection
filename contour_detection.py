import streamlit as st
from PIL import Image
import numpy as np
import cv2

# --------- Contour Detection Function ----------
def detect_contours(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    return contours, edges


def draw_contours(image, contours):
    annotated_image = image.copy()

    for contour in contours:
        # Ignore very small contours
        if cv2.contourArea(contour) > 500:
            cv2.drawContours(
                annotated_image,
                [contour],
                -1,
                (0, 255, 0),
                2
            )

    return annotated_image


# --------- Streamlit App ----------
st.title("Contour Detection App")
st.write("Upload an image to detect object contours.")

uploaded_file = st.file_uploader(
    "Choose an image...", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    st.write("Detecting contours...")

    contours, edges = detect_contours(image_np)
    annotated_image = draw_contours(image_np, contours)

    st.subheader("Detected Contours")
    st.image(annotated_image, use_column_width=True)

    st.subheader("Edge Detection Output")
    st.image(edges, use_column_width=True, clamp=True)