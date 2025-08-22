import streamlit as st
from transformers import pipeline
from PIL import Image

st.title("Image Description App with Hugging Face")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Load Hugging Face multimodal pipeline (e.g., BLIP)
    descriptor = pipeline(
        "image-to-text", 
        model="Salesforce/blip-image-captioning-base"
    )
    description = descriptor(image)[0]['generated_text']
    st.markdown(f"**Description:** {description}")

    # Add additional prompts or tasks here if desired

# Instructions
st.write("This app generates a description for your uploaded image using a Hugging Face multimodal model (BLIP).")

# To run, save this to app.py and use
# streamlit run app.py

# Required packages:
# pip install streamlit transformers Pillow
