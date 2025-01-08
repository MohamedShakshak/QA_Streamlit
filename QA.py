import streamlit as st
from transformers import pipeline
from PIL import Image
import requests
import os 
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Load the pre-trained Visual Question Answering (VQA) model
@st.cache_resource
def load_pipeline():
    return pipeline("visual-question-answering", model="Salesforce/blip-vqa-base")

pipe = load_pipeline()

# Streamlit App
st.title("Visual Question Answering App")
st.write("Upload an image or provide a URL, then ask a question about it!")

# File uploader for image
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

# URL input for image
image_url = st.text_input("Or enter an image URL:")

image = None

# Handle uploaded file
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

# Handle URL input
elif image_url:
    try:
        response = requests.get(image_url, stream=True)
        response.raise_for_status()  # Check if the URL is valid
        image = Image.open(response.raw).convert("RGB")
        st.image(image, caption="Image from URL", use_column_width=True)
    except Exception as e:
        st.error(f"Failed to load image from URL: {e}")

# Input question
if image:
    question = st.text_input("Ask a question about the image:")

    if st.button("Get Answer"):
        if question.strip():
            # Get the model's answer
            with st.spinner("Analyzing..."):
                result = pipe(image, question)
            st.success("Done!")
            st.write(f"*Question:* {question}")
            st.write(f"*Answer:* {result[0]['answer']}")
        else:
            st.warning("Please enter a question.")
else:
    st.info("Please upload an image or enter a valid URL to proceed.")

st.write("---")
st.write("Powered by [Hugging Face](https://huggingface.co/) and [Streamlit](https://streamlit.io/)")
