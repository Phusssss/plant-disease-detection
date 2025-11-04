import streamlit as st
import cv2
import numpy as np
from PIL import Image
from model import PlantDiseaseModel
import os
import gdown

# Download model từ Google Drive nếu chưa có
MODEL_URL = "https://drive.google.com/uc?id=YOUR_MODEL_FILE_ID"
MODEL_PATH = "plant_disease_model.h5"

@st.cache_resource
def download_and_load_model():
    if not os.path.exists(MODEL_PATH):
        st.info("Đang tải model... Vui lòng đợi")
        try:
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        except:
            st.error("Không thể tải model. Vui lòng liên hệ admin.")
            return None
    
    model = PlantDiseaseModel()
    model.load_model(MODEL_PATH)
    return model

def main():
    st.title("🌱 Hệ thống nhận diện bệnh cây trồng")
    
    model = download_and_load_model()
    if model is None:
        return
    
    uploaded_file = st.file_uploader("Tải lên hình ảnh lá cây", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, width=300)
        
        temp_path = "temp.jpg"
        image.save(temp_path)
        
        result = model.predict(temp_path)
        
        plant = result['class'].split('___')[0].replace('_', ' ')
        disease = result['disease'].replace('_', ' ')
        confidence = result['confidence']
        
        st.success(f"🌱 {plant}")
        if disease.lower() == 'healthy':
            st.success("✅ Cây khỏe mạnh")
        else:
            st.warning(f"⚠️ Bệnh: {disease}")
        st.metric("Độ tin cậy", f"{confidence:.1%}")
        
        os.remove(temp_path)

if __name__ == "__main__":
    main()