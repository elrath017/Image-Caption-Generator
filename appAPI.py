from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input, VGG16 #type: ignore
from tensorflow.keras.preprocessing.image import load_img, img_to_array #type: ignore
from tensorflow.keras.models import load_model #type: ignore
from tensorflow.keras.models import model_from_json #type: ignore
from tensorflow.keras.utils import get_custom_objects #type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences #type: ignore
import pickle
import os
from tensorflow import math
import uvicorn
import openai
from pydantic import BaseModel

# Add OpenAI API key
openai.api_key = os.getenv("OPEN_API_KEY")

app = FastAPI()

# Allow cross-origin requests if needed for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model architecture from the JSON file
with open('model_arch.json', 'r') as json_file:
    json_model = json_file.read()
model = model_from_json(json_model)

# Load the model weights
model.load_weights('model_weights.weights.h5')

# print(model.summary())

# Load pre-trained VGG16 for feature extraction
feature_extractor = VGG16()
feature_extractor = tf.keras.Model(inputs=feature_extractor.inputs, outputs=feature_extractor.layers[-2].output)

# Load tokenizer
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
print(tokenizer)

# Parameters
max_length = 34  # Use the max_length from your training code
vocab_size = len(tokenizer.word_index) + 1  # Same as training vocab size

# Function to extract features from an image
def extract_features(image_path):
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return feature_extractor.predict(image, verbose=0)

# Function to generate captions
def generate_caption(photo_feature):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length, padding='post')
        yhat = model.predict([photo_feature, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat, None)
        if word is None or word == 'endseq':
            break
        in_text += ' ' + word
    return in_text.replace('startseq ', '').replace(' endseq', '')

class CaptionRequest(BaseModel):
    caption: str

@app.post('/predict')
async def predict(image: UploadFile = File(...)):
    if not image:
        raise HTTPException(status_code=400, detail="No image file provided")
    
    # Save the image temporarily
    temp_image_path = os.path.join('temp_image.jpg')
    with open(temp_image_path, "wb") as buffer:
        buffer.write(image.file.read())
    
    # Extract features from the uploaded image
    photo_feature = extract_features(temp_image_path).reshape(1, -1)
    
    # Generate caption
    caption = generate_caption(photo_feature)
    
    # Delete the temporary image
    os.remove(temp_image_path)
    
    return {"caption": caption}

@app.post('/generate-story')
async def generate_story(request: CaptionRequest):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a creative storyteller that generates engaging stories based on image captions."},
                {"role": "user", "content": f"Create a short story inspired by this image caption: {request.caption}"}
            ],
            max_tokens=250,
            temperature=0.8
        )
        story = response.choices[0].message.content
        return {"story": story}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)