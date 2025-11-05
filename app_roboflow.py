import streamlit as st
from inference_sdk import InferenceHTTPClient
from PIL import Image
import io

# Roboflow client
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="y0YKSebPyue0doYszJEU"
)

def predict_with_roboflow(image):
    """Dự đoán bằng Roboflow API"""
    # Convert PIL to bytes
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='JPEG')
    img_bytes = img_bytes.getvalue()
    
    # Call API
    result = CLIENT.infer(img_bytes, model_id="plantvillage-dataset/1")
    return result

def format_prediction(result):
    """Format kết quả từ Roboflow"""
    if 'predictions' in result and result['predictions']:
        pred = result['predictions'][0]
        class_name = pred['class']
        confidence = pred['confidence']
        return class_name, confidence
    return "Unknown", 0.0

def main():
    st.set_page_config(
        page_title="🌱 Plant Disease Detection (Roboflow)",
        page_icon="🌱"
    )
    
    st.title("🌱 Nhận diện bệnh cây trồng")
    st.markdown("*Sử dụng Roboflow API - Test nhanh*")
    
    # Upload
    uploaded_file = st.file_uploader(
        "Upload ảnh lá cây",
        type=['jpg', 'jpeg', 'png']
    )
    
    if uploaded_file:
        col1, col2 = st.columns(2)
        
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption="Ảnh upload", use_column_width=True)
        
        with col2:
            if st.button("🔍 Phân tích"):
                with st.spinner("🤖 Đang phân tích..."):
                    try:
                        result = predict_with_roboflow(image)
                        disease, confidence = format_prediction(result)
                        
                        st.success("✅ Hoàn thành!")
                        st.metric("Kết quả", disease, f"{confidence:.1%}")
                        st.progress(confidence)
                        
                        # Show raw result
                        with st.expander("Raw API Response"):
                            st.json(result)
                            
                    except Exception as e:
                        st.error(f"❌ Lỗi: {str(e)}")

if __name__ == "__main__":
    main()