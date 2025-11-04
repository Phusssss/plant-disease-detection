import streamlit as st
import cv2
import numpy as np
from PIL import Image
from model import PlantDiseaseModel
import os

# Cấu hình trang
st.set_page_config(
    page_title="Hệ thống nhận diện bệnh cây trồng",
    page_icon="🌱",
    layout="wide"
)

@st.cache_resource
def load_model():
    model = PlantDiseaseModel()
    if os.path.exists('plant_disease_model.h5'):
        model.load_model('plant_disease_model.h5')
        return model
    return None

def main():
    st.title("🌱 Hệ thống nhận diện bệnh cây trồng")
    st.markdown("---")
    
    # Load model
    model = load_model()
    
    if model is None:
        st.error("⚠️ Không tìm thấy model đã train. Vui lòng train model trước!")
        st.info("Chạy lệnh: `python train.py` để train model")
        return
    
    # Sidebar
    st.sidebar.header("Hướng dẫn sử dụng")
    st.sidebar.markdown("""
    1. Tải lên hình ảnh lá cây
    2. Hệ thống sẽ phân tích và đưa ra kết quả
    3. Xem thông tin chi tiết về bệnh (nếu có)
    """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Tải lên hình ảnh")
        uploaded_file = st.file_uploader(
            "Chọn hình ảnh lá cây",
            type=['jpg', 'jpeg', 'png'],
            help="Hỗ trợ định dạng: JPG, JPEG, PNG"
        )
        
        if uploaded_file is not None:
            # Hiển thị hình ảnh
            image = Image.open(uploaded_file)
            st.image(image, caption="Hình ảnh đã tải lên", use_column_width=True)
            
            # Lưu tạm thời để xử lý
            temp_path = "temp_image.jpg"
            image.save(temp_path)
    
    with col2:
        st.header("🔍 Kết quả phân tích")
        
        if uploaded_file is not None:
            with st.spinner("Đang phân tích hình ảnh..."):
                try:
                    # Dự đoán
                    result = model.predict(temp_path)
                    
                    # Hiển thị kết quả
                    plant_name = result['class'].split('___')[0].replace('_', ' ')
                    disease_name = result['disease'].replace('_', ' ')
                    confidence = result['confidence']
                    
                    st.success("✅ Phân tích hoàn thành!")
                    
                    # Thông tin chi tiết
                    st.subheader("📊 Thông tin chi tiết")
                    
                    col2_1, col2_2 = st.columns(2)
                    with col2_1:
                        st.metric("Loại cây", plant_name)
                        st.metric("Độ tin cậy", f"{confidence:.2%}")
                    
                    with col2_2:
                        if disease_name.lower() == 'healthy':
                            st.success("🌿 Cây khỏe mạnh")
                        else:
                            st.warning(f"⚠️ Bệnh: {disease_name}")
                    
                    # Progress bar cho độ tin cậy
                    st.subheader("📈 Độ tin cậy")
                    st.progress(confidence)
                    
                    # Khuyến nghị
                    st.subheader("💡 Khuyến nghị")
                    if disease_name.lower() == 'healthy':
                        st.info("Cây của bạn trông khỏe mạnh! Tiếp tục chăm sóc tốt.")
                    else:
                        st.warning(f"Cây có thể bị {disease_name}. Nên tham khảo ý kiến chuyên gia nông nghiệp.")
                    
                    # Xóa file tạm
                    os.remove(temp_path)
                    
                except Exception as e:
                    st.error(f"❌ Lỗi khi phân tích: {str(e)}")
        else:
            st.info("👆 Vui lòng tải lên hình ảnh để bắt đầu phân tích")

if __name__ == "__main__":
    main()