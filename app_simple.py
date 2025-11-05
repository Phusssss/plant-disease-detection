import streamlit as st
import requests
from PIL import Image
import io
import base64

def call_roboflow_api(image):
    """Gọi Roboflow API trực tiếp"""
    # Convert image to bytes
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_bytes = buffered.getvalue()
    
    # API call với format đúng
    url = "https://detect.roboflow.com/plantvillage-dataset/1"
    params = {
        "api_key": "y0YKSebPyue0doYszJEU"
    }
    
    response = requests.post(url, params=params, files={"file": img_bytes})
    return response.json()

def main():
    st.set_page_config(
        page_title="🌱 Plant Disease Detection",
        page_icon="🌱"
    )
    
    st.title("🌱 Nhận diện bệnh cây trồng")
    st.markdown("*Test với Roboflow API*")
    
    uploaded_file = st.file_uploader(
        "Upload ảnh lá cây",
        type=['jpg', 'jpeg', 'png']
    )
    
    if uploaded_file:
        col1, col2 = st.columns(2)
        
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption="Ảnh upload")
        
        with col2:
            if st.button("🔍 Phân tích"):
                with st.spinner("Đang phân tích..."):
                    try:
                        result = call_roboflow_api(image)
                        
                        if 'predictions' in result and result['predictions']:
                            pred = result['predictions'][0]
                            disease = pred['class']
                            confidence = pred['confidence']
                            
                            st.success("✅ Hoàn thành!")
                            st.metric("Kết quả", disease, f"{confidence:.1%}")
                            st.progress(confidence)
                        else:
                            st.warning("Không phát hiện bệnh")
                        
                        with st.expander("Raw Response"):
                            st.json(result)
                            
                    except Exception as e:
                        st.error(f"Lỗi: {str(e)}")

if __name__ == "__main__":
    main()