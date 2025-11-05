import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
import os
from io import BytesIO

# Class names
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

@st.cache_resource
def load_model():
    """Load model từ file hoặc URL"""
    model_path = "plant_disease_model.h5"
    
    if not os.path.exists(model_path):
        st.warning("⬇️ Đang tải model...")
        # Thay URL này bằng link Google Drive/AWS S3 của bạn
        model_url = "YOUR_MODEL_URL_HERE"
        
        try:
            response = requests.get(model_url)
            with open(model_path, 'wb') as f:
                f.write(response.content)
            st.success("✅ Model đã được tải!")
        except:
            st.error("❌ Không thể tải model. Vui lòng kiểm tra URL.")
            return None
    
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except:
        st.error("❌ Lỗi khi load model")
        return None

def preprocess_image(image):
    """Tiền xử lý ảnh"""
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def predict_disease(model, image):
    """Dự đoán bệnh"""
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    
    return CLASS_NAMES[predicted_class], confidence

def format_disease_name(class_name):
    """Format tên bệnh cho dễ đọc"""
    parts = class_name.split('___')
    plant = parts[0].replace('_', ' ')
    disease = parts[1].replace('_', ' ') if len(parts) > 1 else 'Unknown'
    
    if disease.lower() == 'healthy':
        return f"🌱 {plant} - Khỏe mạnh"
    else:
        return f"🦠 {plant} - {disease}"

def main():
    st.set_page_config(
        page_title="🌱 Nhận diện bệnh cây trồng",
        page_icon="🌱",
        layout="wide"
    )
    
    st.title("🌱 Hệ thống nhận diện bệnh cây trồng")
    st.markdown("*Sử dụng AI để nhận diện bệnh trên lá cây*")
    
    # Load model
    model = load_model()
    if model is None:
        st.stop()
    
    # Sidebar
    st.sidebar.header("📋 Hướng dẫn")
    st.sidebar.markdown("""
    1. Upload ảnh lá cây
    2. Chờ AI phân tích
    3. Xem kết quả và độ tin cậy
    
    **Hỗ trợ 14 loại cây:**
    - Táo, Việt quất, Anh đào
    - Ngô, Nho, Cam, Đào
    - Ớt chuông, Khoai tây
    - Mâm xôi, Đậu nành, Bí
    - Dâu tây, Cà chua
    """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload ảnh")
        uploaded_file = st.file_uploader(
            "Chọn ảnh lá cây",
            type=['jpg', 'jpeg', 'png'],
            help="Hỗ trợ JPG, JPEG, PNG"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Ảnh đã upload", use_column_width=True)
    
    with col2:
        st.header("🔍 Kết quả")
        
        if uploaded_file is not None:
            with st.spinner("🤖 AI đang phân tích..."):
                try:
                    disease, confidence = predict_disease(model, image)
                    formatted_disease = format_disease_name(disease)
                    
                    # Hiển thị kết quả
                    st.success("✅ Phân tích hoàn thành!")
                    
                    st.metric(
                        label="Kết quả",
                        value=formatted_disease,
                        delta=f"Độ tin cậy: {confidence:.1%}"
                    )
                    
                    # Progress bar cho confidence
                    st.progress(confidence)
                    
                    # Thông tin thêm
                    if confidence > 0.8:
                        st.info("🎯 Kết quả có độ tin cậy cao")
                    elif confidence > 0.6:
                        st.warning("⚠️ Kết quả có độ tin cậy trung bình")
                    else:
                        st.error("❌ Kết quả có độ tin cậy thấp")
                    
                except Exception as e:
                    st.error(f"❌ Lỗi khi phân tích: {str(e)}")
        else:
            st.info("👆 Vui lòng upload ảnh để bắt đầu")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "🚀 **Powered by TensorFlow & Streamlit** | "
        "🌱 **Plant Disease Detection System**"
    )

if __name__ == "__main__":
    main()