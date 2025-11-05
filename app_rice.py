import streamlit as st
import requests
from PIL import Image
import io

def call_rice_api(image):
    """Gọi API chẩn đoán bệnh lúa"""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_bytes = buffered.getvalue()
    
    url = "https://detect.roboflow.com/rice-diseases-qzjka/3"
    params = {"api_key": "y0YKSebPyue0doYszJEU"}
    
    response = requests.post(url, params=params, files={"file": img_bytes})
    return response.json()

def format_rice_disease(class_name):
    """Format tên bệnh lúa"""
    disease_map = {
        'bacterial_leaf_blight': '🦠 Bệnh cháy lá do vi khuẩn',
        'brown_spot': '🟤 Bệnh đốm nâu',
        'leaf_smut': '🖤 Bệnh than lá',
        'healthy': '🌱 Lúa khỏe mạnh'
    }
    return disease_map.get(class_name.lower(), f"🌾 {class_name}")

def main():
    st.set_page_config(
        page_title="🌾 Chẩn đoán bệnh lúa",
        page_icon="🌾"
    )
    
    st.title("🌾 Hệ thống chẩn đoán bệnh cây lúa")
    st.markdown("*Chuyên dụng cho cây lúa - Powered by Roboflow*")
    
    # Thông tin bệnh
    st.sidebar.header("📋 Các bệnh phổ biến")
    st.sidebar.markdown("""
    **Bệnh thường gặp:**
    - 🦠 Bệnh cháy lá do vi khuẩn
    - 🟤 Bệnh đốm nâu  
    - 🖤 Bệnh than lá
    - 🌱 Lúa khỏe mạnh
    
    **Hướng dẫn:**
    1. Chụp ảnh lá lúa rõ nét
    2. Upload ảnh vào hệ thống
    3. Nhận kết quả chẩn đoán
    """)
    
    uploaded_file = st.file_uploader(
        "📤 Upload ảnh lá lúa",
        type=['jpg', 'jpeg', 'png'],
        help="Chụp ảnh lá lúa rõ nét để có kết quả chính xác nhất"
    )
    
    if uploaded_file:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption="Ảnh lá lúa", use_column_width=True)
        
        with col2:
            st.header("🔍 Kết quả chẩn đoán")
            
            if st.button("🌾 Chẩn đoán bệnh lúa", type="primary"):
                with st.spinner("🤖 AI đang phân tích lá lúa..."):
                    try:
                        result = call_rice_api(image)
                        
                        if 'predictions' in result and result['predictions']:
                            # Lấy prediction có confidence cao nhất
                            best_pred = max(result['predictions'], key=lambda x: x['confidence'])
                            
                            disease = best_pred['class']
                            confidence = best_pred['confidence']
                            formatted_disease = format_rice_disease(disease)
                            
                            st.success("✅ Chẩn đoán hoàn thành!")
                            
                            # Hiển thị kết quả
                            st.metric(
                                label="Kết quả",
                                value=formatted_disease,
                                delta=f"Độ tin cậy: {confidence:.1%}"
                            )
                            
                            # Progress bar
                            st.progress(confidence)
                            
                            # Đánh giá độ tin cậy
                            if confidence > 0.8:
                                st.info("🎯 Kết quả có độ tin cậy cao")
                            elif confidence > 0.6:
                                st.warning("⚠️ Kết quả có độ tin cậy trung bình")
                            else:
                                st.error("❌ Kết quả có độ tin cậy thấp - Nên chụp ảnh rõ hơn")
                            
                            # Hiển thị tất cả predictions
                            if len(result['predictions']) > 1:
                                with st.expander("📊 Chi tiết các khả năng"):
                                    for pred in sorted(result['predictions'], key=lambda x: x['confidence'], reverse=True):
                                        st.write(f"• {format_rice_disease(pred['class'])}: {pred['confidence']:.1%}")
                        
                        else:
                            st.warning("⚠️ Không phát hiện bệnh hoặc ảnh không rõ")
                            st.info("💡 Thử chụp ảnh lá lúa rõ nét hơn")
                        
                        # Raw response cho debug
                        with st.expander("🔧 Raw API Response"):
                            st.json(result)
                            
                    except Exception as e:
                        st.error(f"❌ Lỗi khi chẩn đoán: {str(e)}")
                        st.info("💡 Kiểm tra kết nối internet và thử lại")
    
    else:
        st.info("👆 Vui lòng upload ảnh lá lúa để bắt đầu chẩn đoán")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "🌾 **Rice Disease Detection System** | "
        "🚀 **Powered by Roboflow AI** | "
        "🎯 **Chuyên dụng cho cây lúa**"
    )

if __name__ == "__main__":
    main()