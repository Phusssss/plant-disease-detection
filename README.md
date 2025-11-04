# 🌱 Hệ thống nhận diện bệnh cây trồng

Hệ thống AI sử dụng deep learning để nhận diện các bệnh phổ biến trên cây trồng thông qua hình ảnh lá.

## ✨ Tính năng

- Nhận diện 38 loại bệnh trên 14 loại cây trồng khác nhau
- Giao diện web thân thiện với Streamlit
- Độ chính xác cao với CNN
- Hỗ trợ nhiều định dạng ảnh (JPG, PNG, JPEG)

## 🚀 Cài đặt

1. Clone repository:
```bash
git clone <repository-url>
cd plant_disease_detection
```

2. Cài đặt dependencies:
```bash
pip install -r requirements.txt
```

3. Tải dataset PlantVillage và đặt vào thư mục `data/PlantVillage/`

## 📊 Sử dụng

### Train model:
```bash
python train.py
```

### Chạy ứng dụng web:
```bash
streamlit run app.py
```

## 🌿 Các loại cây được hỗ trợ

- Táo (Apple)
- Việt quất (Blueberry) 
- Anh đào (Cherry)
- Ngô (Corn)
- Nho (Grape)
- Cam (Orange)
- Đào (Peach)
- Ớt chuông (Bell Pepper)
- Khoai tây (Potato)
- Mâm xôi (Raspberry)
- Đậu nành (Soybean)
- Bí (Squash)
- Dâu tây (Strawberry)
- Cà chua (Tomato)

## 🔧 Cấu trúc dự án

```
plant_disease_detection/
├── model.py          # Class chính cho model
├── train.py          # Script train model
├── app.py            # Ứng dụng Streamlit
├── requirements.txt  # Dependencies
└── README.md         # Hướng dẫn
```

## 📈 Hiệu suất

Model sử dụng CNN với các layer:
- Conv2D + MaxPooling
- Dropout để tránh overfitting
- Dense layers cho classification
- Accuracy: ~95% trên validation set

## 🌐 Deployment

### Cách deploy lên hosting:

1. **Upload chỉ những file cần thiết:**
   - `model.py`
   - `deploy.py` (thay cho app.py)
   - `requirements_deploy.txt`
   - `plant_disease_model.h5` (model đã train)

2. **KHÔNG cần upload:**
   - Dataset training (thư mục data/)
   - Hình ảnh training

3. **Chạy trên hosting:**
```bash
pip install -r requirements_deploy.txt
streamlit run deploy.py
```

### Lưu trữ model:
- Upload model lên Google Drive/AWS S3
- App sẽ tự động download khi cần