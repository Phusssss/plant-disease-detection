from model import PlantDiseaseModel
import sys
import os

def predict_image(image_path):
    # Load model
    model = PlantDiseaseModel()
    
    if not os.path.exists('plant_disease_model.h5'):
        print("❌ Không tìm thấy model đã train!")
        print("Vui lòng chạy: python train.py")
        return
    
    model.load_model('plant_disease_model.h5')
    
    # Dự đoán
    try:
        result = model.predict(image_path)
        
        plant_name = result['class'].split('___')[0].replace('_', ' ')
        disease_name = result['disease'].replace('_', ' ')
        confidence = result['confidence']
        
        print(f"🌱 Loại cây: {plant_name}")
        print(f"🔍 Tình trạng: {disease_name}")
        print(f"📊 Độ tin cậy: {confidence:.2%}")
        
        if disease_name.lower() == 'healthy':
            print("✅ Cây khỏe mạnh!")
        else:
            print(f"⚠️  Phát hiện bệnh: {disease_name}")
            
    except Exception as e:
        print(f"❌ Lỗi: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Sử dụng: python predict_single.py <đường_dẫn_ảnh>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"❌ Không tìm thấy file: {image_path}")
        sys.exit(1)
    
    predict_image(image_path)