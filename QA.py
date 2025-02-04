import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing import sequence
from PIL import Image
import requests
import matplotlib.pyplot as plt
import os
import pickle

# Load Models and Tokenizers
@st.cache_resource
def load_models_and_tokenizers():
    fe_model = load_model("inceptionV3_features_model.h5")
    caption_model = load_model("model50.keras")
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

def beam_search_predictions(image, beam_index = 3):
    start = [wordtoix["startseq"]]
    start_word = [[start, 0.0]]
    while len(start_word[0][0]) < max_length:
        temp = []
        for s in start_word:
            par_caps = sequence.pad_sequences([s[0]], maxlen=max_length, padding='post')
            preds = caption_model.predict([image,par_caps], verbose=0)
            word_preds = np.argsort(preds[0])[-beam_index:]
            # Getting the top <beam_index>(n) predictions and creating a 
            # new list so as to put them via the model again
            for w in word_preds:
                next_cap, prob = s[0][:], s[1]
                next_cap.append(w)
                prob += preds[0][w]
                temp.append([next_cap, prob])
                    
        start_word = temp
        # Sorting according to the probabilities
        start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
        # Getting the top words
        start_word = start_word[-beam_index:]
    
    start_word = start_word[-1][0]
    intermediate_caption = [ixtoword[i] for i in start_word]
    final_caption = []
    
    for i in intermediate_caption:
        if i != 'endseq':
            final_caption.append(i)
        else:
            break

    final_caption = ' '.join(final_caption[1:])
    return final_caption


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
        st.write(f"**Beam Search, K = 3:** {beam_search_predictions(encoded_image, beam_index = 3)}")
        st.write(f"**Beam Search, K = 5:** {beam_search_predictions(encoded_image, beam_index = 5)}")
        st.write(f"**Beam Search, K = 7:** {beam_search_predictions(encoded_image, beam_index = 7)}")
else:
    st.info("Please upload an image or provide a valid URL to proceed.")

st.write("---")
st.write("Powered by [Streamlit](https://streamlit.io/) and TensorFlow")
