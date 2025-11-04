import os
import zipfile



def download_dataset():
    print("Tai dataset PlantVillage...")
    
    # Tạo thư mục data nếu chưa có
    os.makedirs("data", exist_ok=True)
    
    # URL dataset (Kaggle PlantVillage)
    dataset_url = "https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset/download"
    
    print("Huong dan tai dataset:")
    print("1. Truy cap: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset")
    print("2. Dang nhap Kaggle va tai file zip")
    print("3. Giai nen vao thu muc data/PlantVillage/")
    print("4. Hoac su dung Kaggle API:")
    print("   kaggle datasets download -d abdallahalidev/plantvillage-dataset")
    print("   unzip plantvillage-dataset.zip -d data/")
    
    # Tạo dataset mẫu nhỏ để test
    print("\nTao dataset mau de test...")
    sample_dir = "data/PlantVillage_sample"
    os.makedirs(f"{sample_dir}/Tomato___healthy", exist_ok=True)
    os.makedirs(f"{sample_dir}/Tomato___Late_blight", exist_ok=True)
    
    print("Da tao cau truc thu muc mau")
    print("Cau truc:")
    print("   data/PlantVillage_sample/")
    print("   |-- Tomato___healthy/")
    print("   |-- Tomato___Late_blight/")
    print("\nThem vai anh vao cac thu muc nay de test!")

if __name__ == "__main__":
    download_dataset()