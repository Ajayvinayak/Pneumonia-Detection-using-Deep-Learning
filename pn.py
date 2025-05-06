import warnings
import base64
import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# ✅ Load trained model
model = tf.keras.models.load_model("pneumonia_mobilenetv2_finetuned.h5")

# ✅ Function to add a background image with transparency
def add_bg_from_local(image_file, opacity=0.6):  # Set opacity to 60%
    with open(image_file, "rb") as image_file:
        b64_string = base64.b64encode(image_file.read()).decode()

    bg_image = f"""
    <style>
    .stApp {{
        background: linear-gradient(rgba(0, 0, 0, {opacity}), rgba(0, 0, 0, {opacity})), 
                    url("data:image/png;base64,{b64_string}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: white;  /* Ensures all text is white */
    }}
    .stMarkdown, .stTitle, .stText, .stHeader, .stSubheader {{
        color: white !important;  /* Forces all text to be white */
    }}
    </style>
    """
    st.markdown(bg_image, unsafe_allow_html=True)

# ✅ Apply background image (Ensure the file exists)
add_bg_from_local("ai pn.webp", opacity=0.6)  # Adjust opacity if needed

# ✅ Streamlit UI
st.title("🩺 Pneumonia Detection from Chest X-ray")
st.write("Upload a chest X-ray image to check for Pneumonia.")

# ✅ Upload file
uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "png", "jpeg"])

def preprocess_image(img):
    img = image.load_img(img, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

if uploaded_file is not None:
    # Display image without the deprecation warning
    st.image(uploaded_file, caption="Uploaded X-ray", use_container_width=True)

    # Preprocess image
    img_array = preprocess_image(uploaded_file)

    # Make prediction
    prediction = model.predict(img_array)[0][0]
    class_label = "Pneumonia" if prediction > 0.5 else "Normal"
    confidence = float(prediction) if prediction > 0.5 else 1 - float(prediction)

    # Display results
    st.write(f"✅ **Prediction:** {class_label}")
    st.write(f"📌 **Confidence Score:** {confidence:.2f}")

# ✅ Hide Streamlit warnings
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_option('deprecation.showUseColumnWidth', False)

# ✅ Hide all warnings (Optional)
warnings.filterwarnings("ignore")
