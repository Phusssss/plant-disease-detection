import streamlit as st
import requests
from PIL import Image
import io

def call_plant_api(image):
    """API chẩn đoán bệnh cây trồng tổng quát"""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_bytes = buffered.getvalue()
    
    url = "https://detect.roboflow.com/plantvillage-dataset/1"
    params = {"api_key": "y0YKSebPyue0doYszJEU"}
    
    response = requests.post(url, params=params, files={"file": img_bytes})
    return response.json()

def call_rice_api(image):
    """API chẩn đoán bệnh lúa chuyên dụng"""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_bytes = buffered.getvalue()
    
    url = "https://detect.roboflow.com/rice-diseases-qzjka/3"
    params = {"api_key": "y0YKSebPyue0doYszJEU"}
    
    response = requests.post(url, params=params, files={"file": img_bytes})
    return response.json()

def format_rice_disease(class_name):
    """Format tên bệnh lúa sang tiếng Việt"""
    disease_map = {
        'bacterial leaf blight or bacterial blight disease': '🦠 Bệnh cháy lá do vi khuẩn',
        'bacterial leaf streak disease': '🦠 Bệnh vằn lá do vi khuẩn',
        'brown spot disease': '🟤 Bệnh đốm nâu',
        'dirty panicle disease': '🟫 Bệnh bông bẩn',
        'grassy stunt disease': '🌿 Bệnh lùn cỏ',
        'narrow brown spot disease': '🟤 Bệnh đốm nâu hẹp',
        'ragged stunt disease': '🍂 Bệnh lùn rách',
        'rice blast disease': '💥 Bệnh đạo ôn',
        'sheath blight disease': '🟨 Bệnh khô vỏ lá',
        'tungro disease or yellow orange leaf disease': '🟡 Bệnh tungro (lá vàng cam)'
    }
    return disease_map.get(class_name.lower(), f"🌾 {class_name}")

def main():
    st.set_page_config(
        page_title="🌱 Hệ thống chẩn đoán bệnh cây trồng",
        page_icon="🌱",
        layout="wide"
    )
    
    st.title("🌱 Hệ thống chẩn đoán bệnh cây trồng")
    st.markdown("*AI chẩn đoán bệnh cho nhiều loại cây trồng - Hỗ trợ 14 loại cây + 10 bệnh lúa*")
    
    # Tạo 2 tabs
    tab1, tab2 = st.tabs(["🌿 Cây trồng tổng quát (14 loại)", "🌾 Cây lúa chuyên dụng (10 bệnh)"])
    
    # TAB 1: Cây trồng tổng quát
    with tab1:
        st.header("🌿 Chẩn đoán bệnh cây trồng")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("**Các loại cây được hỗ trợ:**")
            st.markdown("""
            - 🍎 Táo, 🫐 Việt quất, 🍒 Anh đào
            - 🌽 Ngô, 🍇 Nho, 🍊 Cam, 🍑 Đào
            - 🫑 Ớt chuông, 🥔 Khoai tây
            - 🫐 Mâm xôi, 🌱 Đậu nành, 🎃 Bí
            - 🍓 Dâu tây, 🍅 Cà chua
            """)
            
            uploaded_file1 = st.file_uploader(
                "📤 Upload ảnh lá cây",
                type=['jpg', 'jpeg', 'png'],
                key="plant_upload"
            )
            
            if uploaded_file1:
                image1 = Image.open(uploaded_file1)
                st.image(image1, caption="Ảnh đã upload")
        
        with col2:
            st.markdown("**🔍 Kết quả chẩn đoán:**")
            
            if uploaded_file1:
                if st.button("🌿 Chẩn đoán bệnh cây", key="plant_btn"):
                    with st.spinner("🤖 Đang phân tích..."):
                        try:
                            result = call_plant_api(image1)
                            
                            if 'predictions' in result and result['predictions']:
                                pred = result['predictions'][0]
                                disease = pred['class']
                                confidence = pred['confidence']
                                
                                st.success("✅ Hoàn thành!")
                                st.metric("Kết quả", disease, f"{confidence:.1%}")
                                st.progress(confidence)
                                
                                if confidence > 0.8:
                                    st.info("🎯 Độ tin cậy cao")
                                elif confidence > 0.6:
                                    st.warning("⚠️ Độ tin cậy trung bình")
                                else:
                                    st.error("❌ Độ tin cậy thấp")
                            else:
                                st.warning("Không phát hiện bệnh")
                            
                            with st.expander("Raw Response"):
                                st.json(result)
                                
                        except Exception as e:
                            st.error(f"Lỗi: {str(e)}")
            else:
                st.info("👆 Upload ảnh để bắt đầu")
    
    # TAB 2: Cây lúa chuyên dụng
    with tab2:
        st.header("🌾 Chẩn đoán bệnh cây lúa")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("**Các bệnh lúa được hỗ trợ (10 loại):**")
            st.markdown("""
            - 🦠 **Bệnh cháy lá do vi khuẩn**
            - 🦠 **Bệnh vằn lá do vi khuẩn**
            - 🟤 **Bệnh đốm nâu**
            - 🟫 **Bệnh bông bẩn**
            - 🌿 **Bệnh lùn cỏ**
            - 🟤 **Bệnh đốm nâu hẹp**
            - 🍂 **Bệnh lùn rách**
            - 💥 **Bệnh đạo ôn**
            - 🟨 **Bệnh khô vỏ lá**
            - 🟡 **Bệnh tungro (lá vàng cam)**
            
            💡 *Chụp ảnh lá lúa rõ nét để có kết quả chính xác*
            """)
            
            uploaded_file2 = st.file_uploader(
                "📤 Upload ảnh lá lúa",
                type=['jpg', 'jpeg', 'png'],
                key="rice_upload"
            )
            
            if uploaded_file2:
                image2 = Image.open(uploaded_file2)
                st.image(image2, caption="Ảnh lá lúa")
        
        with col2:
            st.markdown("**🔍 Kết quả chẩn đoán lúa:**")
            
            if uploaded_file2:
                if st.button("🌾 Chẩn đoán bệnh lúa", key="rice_btn"):
                    with st.spinner("🤖 Đang phân tích lúa..."):
                        try:
                            result = call_rice_api(image2)
                            
                            if 'predictions' in result and result['predictions']:
                                best_pred = max(result['predictions'], key=lambda x: x['confidence'])
                                disease = best_pred['class']
                                confidence = best_pred['confidence']
                                formatted_disease = format_rice_disease(disease)
                                
                                st.success("✅ Chẩn đoán hoàn thành!")
                                st.metric("Kết quả", formatted_disease, f"{confidence:.1%}")
                                st.progress(confidence)
                                
                                if confidence > 0.8:
                                    st.info("🎯 Độ tin cậy cao")
                                elif confidence > 0.6:
                                    st.warning("⚠️ Độ tin cậy trung bình")
                                else:
                                    st.error("❌ Độ tin cậy thấp")
                                
                                # Hiển thị tất cả predictions
                                if len(result['predictions']) > 1:
                                    with st.expander("📊 Chi tiết"):
                                        for pred in sorted(result['predictions'], key=lambda x: x['confidence'], reverse=True):
                                            st.write(f"• {format_rice_disease(pred['class'])}: {pred['confidence']:.1%}")
                            else:
                                st.warning("Không phát hiện bệnh lúa")
                            
                            with st.expander("Raw Response"):
                                st.json(result)
                                
                        except Exception as e:
                            st.error(f"Lỗi: {str(e)}")
            else:
                st.info("👆 Upload ảnh lúa để bắt đầu")
    
    # Footer
    st.markdown("---")
    st.markdown("🚀 **Powered by Roboflow AI** | 🌱 **Plant Disease Detection System**")

if __name__ == "__main__":
    main()