import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

st.set_page_config(page_title="Dog vs Cat Classifier", page_icon="🐶")
st.title("🐶 Dog vs Cat Classification 🐱")
st.write("Upload a dog or cat photo to classify it using MobileNetV2 transfer learning.")

@st.cache_resource
def load_classifier():
    return tf.keras.models.load_model(Path(__file__).parent / "dog_cat_classifier.keras", compile=False)

try:
    model = load_classifier()
except Exception as exc:
    st.error(f"Could not load the model: {exc}")
    st.stop()

uploaded_file = st.file_uploader("Upload a JPG or PNG image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)
    if st.button("Predict", type="primary"):
        resized = image.resize((224, 224))
        image_array = np.asarray(resized, dtype=np.float32)
        image_array = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)
        image_array = np.expand_dims(image_array, axis=0)
        dog_probability = float(model.predict(image_array, verbose=0)[0][0])
        label = "Dog 🐶" if dog_probability >= 0.5 else "Cat 🐱"
        confidence = dog_probability if dog_probability >= 0.5 else 1 - dog_probability
        st.success(f"Prediction: {label}")
        st.write(f"Model score: {confidence:.2%} (not a guarantee of correctness)")
        st.caption("For images containing both cats and dogs, the model still predicts one class.")
