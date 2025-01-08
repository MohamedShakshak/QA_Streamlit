import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import requests
import matplotlib.pyplot as plt
import os

# Load Models and Tokenizers
@st.cache_resource
def load_models_and_tokenizers():
    fe_model = load_model("inceptionV3_features_model.h5")
    caption_model = load_model("model.keras")
    with open('wordtoix.pkl', 'rb') as file:
        wordtoix = pickle.load(file)
    with open('ixtoword.pkl', 'rb') as file:
        ixtoword = pickle.load(file)
    return fe_model, caption_model, wordtoix, ixtoword

fe_model, caption_model, wordtoix, ixtoword = load_models_and_tokenizers()
max_length = 51

# Preprocessing Functions
def preprocess(image_path):
    img = load_img(image_path, target_size=(299, 299))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def encode(image_path):
    image = preprocess(image_path)
    fea_vec = fe_model.predict(image, verbose=0)
    fea_vec = np.reshape(fea_vec, fea_vec.shape[1])
    return fea_vec

def greedySearch(photo):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = caption_model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = ixtoword[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break
    return ' '.join(in_text.split()[1:-1])

# Streamlit App
st.title("Image Captioning App")
st.write("Upload an image or provide a URL to generate captions.")

# Image Upload or URL Input
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
image_url = st.text_input("Or enter an image URL:")

image = None
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_path = "temp_uploaded_image.jpg"
    image.save(image_path)
    st.image(image, caption="Uploaded Image", use_column_width=True)
elif image_url:
    try:
        response = requests.get(image_url, stream=True)
        response.raise_for_status()
        image = Image.open(response.raw).convert("RGB")
        image_path = "temp_url_image.jpg"
        image.save(image_path)
        st.image(image, caption="Image from URL", use_column_width=True)
    except Exception as e:
        st.error(f"Failed to load image from URL: {e}")

# Generate Captions
if image:
    if st.button("Generate Captions"):
        with st.spinner("Generating captions..."):
            encoded_image = encode(image_path)
            encoded_image = encoded_image.reshape((1, 2048))
            greedy_caption = greedySearch(encoded_image)
        st.success("Captioning Completed!")
        st.write(f"**Greedy Search Caption:** {greedy_caption}")
else:
    st.info("Please upload an image or provide a valid URL to proceed.")

st.write("---")
st.write("Powered by [Streamlit](https://streamlit.io/) and TensorFlow")
