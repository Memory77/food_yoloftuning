import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from ultralytics import YOLO
import av
import cv2

# Charger le modèle YOLOv9 pré-entraîné
model_path = 'best.pt'
model = YOLO(model_path)

# Dictionnaire des calories par aliment
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

        # Effectuer l'inférence
        results = self.model(img)

        # Extraire les noms des classes détectées
        self.detected_items = [self.model.names[int(cls)] for cls in results[0].boxes.cls]

        # Dessiner les boîtes de détection sur l'image
        annotated_img = results[0].plot()

        return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")

    def get_detected_items(self):
        return self.detected_items

st.title("Détection d'aliments en temps réel avec YOLOv9")

webrtc_ctx = webrtc_streamer(key="example", 
                             video_processor_factory=VideoProcessor, 
                             rtc_configuration=RTC_CONFIGURATION)

if webrtc_ctx.video_processor:
    detected_items = webrtc_ctx.video_processor.get_detected_items()
    if st.button("Scanner"):
        total_calories = sum(calories_dict.get(item, 0) for item in detected_items)
        st.write(f"Aliments détectés : {', '.join(detected_items)}")
        st.write(f"Calories totales : {total_calories} kcal")
        st.write(f"add test") 
    
