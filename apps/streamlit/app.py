import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from ultralytics import YOLO
import av
import os
import openai
from dotenv import load_dotenv
import cv2

# Load environment variables from .env file
load_dotenv()

# Configure OpenAI with Azure credentials
openai.api_key = st.secrets["openai_api_key"]
openai.api_base = st.secrets["openai_api_key_base"]
openai.api_type = 'azure'
openai.api_version = st.secrets["openai_api_version"]
api_deployment = st.secrets["openai_api_deployment"]


# Function to call OpenAI API and get a response
def get_response(prompt):
    response = openai.ChatCompletion.create(
        engine=api_deployment,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300
    )
    return response.choices[0].message['content'].strip()

# Load YOLO model
model_path = 'train50ep10m/best.pt'
model = YOLO(model_path)

# Dictionary of calories per food item
calories_dict = {
    'Apple': 52, 'Apricot': 48, 'Aubergine': 25, 'Avocado': 160, 'Banana': 89,
    'Beef Curry': 150, 'Beef Steak': 271, 'Bread': 265, 'Cabbage': 25, 'Carrot': 41,
    'Cauliflower': 25, 'Cheese': 402, 'Cherry': 50, 'Chicken': 239, 'Chili': 40,
    'Corn': 96, 'Croissant': 406, 'Cucumber': 16, 'Dates': 277, 'Egg': 155,
    'Fig': 74, 'Finger': 90, 'French Fries': 312, 'Garlic': 149, 'Grapes': 69,
    'Green Onions': 32, 'Green Salad': 20, 'Hamburger': 295, 'Hot Dog': 151,
    'Kiwi': 61, 'Lemon': 29, 'Lettuce': 15, 'Melon': 30, 'Omelet': 154,
    'Orange': 47, 'Pasta': 131, 'Peach': 39, 'Pear': 57, 'Pineapple': 50,
    'Pizza': 266, 'Pomegranate': 83, 'Potato': 77, 'Rice': 130, 'Sauce': 40,
    'Sausage': 301, 'Strawberry': 32, 'Sushi': 93, 'Tomato': 18, 'Watermelon': 30
}

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model
        self.detected_items = []

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Perform inference
        results = self.model(img)

        # Extract detected class names
        self.detected_items = [self.model.names[int(cls)] for cls in results[0].boxes.cls]

        # Draw detection boxes on the image
        annotated_img = results[0].plot()

        return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")

    def get_detected_items(self):
        return self.detected_items

st.title("Real-Time Food Detection and Recipe Suggestion")

webrtc_ctx = webrtc_streamer(key="example", 
                             video_processor_factory=VideoProcessor, 
                             rtc_configuration=RTC_CONFIGURATION)

if webrtc_ctx.video_processor:
    detected_items = webrtc_ctx.video_processor.get_detected_items()
    if st.button("Scan"):
        total_calories = sum(calories_dict.get(item, 0) for item in detected_items)
        st.write(f"Detected items: {', '.join(detected_items)}")
        st.write(f"Total calories: {total_calories} kcal")

        # Create a prompt for the recipe suggestion
        prompt = f"Je possède ces différents ingrédients : {', '.join(detected_items)}. Peux tu me suggérer différentes recettes ? Ne compte pas finger parmis les ingrédients"
        recipe = get_response(prompt)
        st.write("Suggested recipe:")
        st.write(recipe)