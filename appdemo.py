import streamlit as st
from PIL import Image
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input, VGG16
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

# App layout
st.title("Image Caption Generator")
st.write("Upload an image to generate its caption.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Load the model architecture and weights
@st.cache_resource
def load_captioning_model():
    with open('model_arch.json', 'r') as json_file:
        json_model = json_file.read()
    model = model_from_json(json_model)
    model.load_weights('model_weights.weights.h5')
    return model

# Load pre-trained VGG16 for feature extraction
@st.cache_resource
def load_feature_extractor():
    vgg_model = VGG16()
    return tf.keras.Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)

# Load tokenizer
@st.cache_resource
def load_tokenizer():
    with open('tokenizer.pkl', 'rb') as f:
        return pickle.load(f)

model = load_captioning_model()
feature_extractor = load_feature_extractor()
tokenizer = load_tokenizer()
max_length = 34

# Function to extract features from an image
def extract_features(image):
    image = image.resize((224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return feature_extractor.predict(image, verbose=0)

# Convert integer to word
def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# Generate captions for the image
def predict_caption(model, image, tokenizer, max_length):
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length, padding='post')
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None:
            break
        in_text += " " + word
        if word == 'endseq':
            break
    return in_text.replace('startseq', '').replace('endseq', '').strip()

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        photo_feature = extract_features(image)
        caption = predict_caption(model, photo_feature, tokenizer, max_length)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write(f"**Generated Caption:** {caption}")
    except Exception as e:
        st.error(f"Error generating caption: {e}")
else:
    st.write("Please upload an image to proceed.")
