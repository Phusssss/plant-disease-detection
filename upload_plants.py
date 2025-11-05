import requests
import json
import time

# Thay URL này bằng URL Railway của bạn
API_URL = "https://your-railway-url.railway.app"

def upload_plants():
    """Upload tất cả plants từ file JSON lên API"""
    
    # Đọc dữ liệu từ file
    with open('sample_plants.json', 'r', encoding='utf-8') as f:
        plants_data = json.load(f)
    
    print(f"🌱 Bắt đầu upload {len(plants_data)} cây trồng...")
    
    success_count = 0
    error_count = 0
    
    for i, plant in enumerate(plants_data, 1):
        try:
            response = requests.post(
                f"{API_URL}/plants",
                json=plant,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                success_count += 1
                print(f"✅ {i:3d}. {plant['name']} - Thành công")
            else:
                error_count += 1
                print(f"❌ {i:3d}. {plant['name']} - Lỗi: {response.status_code}")
                print(f"    {response.text}")
                
        except Exception as e:
            error_count += 1
            print(f"❌ {i:3d}. {plant['name']} - Exception: {str(e)}")
        
        # Delay nhỏ để tránh spam API
        time.sleep(0.1)
    
    print(f"\n📊 Kết quả:")
    print(f"✅ Thành công: {success_count}")
    print(f"❌ Lỗi: {error_count}")
    print(f"📈 Tỷ lệ thành công: {success_count/(success_count+error_count)*100:.1f}%")

def test_api():
    """Test API trước khi upload"""
    try:
        # Test root endpoint
        response = requests.get(f"{API_URL}/")
        print(f"Root endpoint: {response.status_code}")
        
        # Test plants endpoint
        response = requests.get(f"{API_URL}/plants")
        print(f"Plants endpoint: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ API có endpoint /plants")
            return True
        else:
            print(f"❌ Endpoint /plants không tồn tại: {response.status_code}")
            print("💡 Cần redeploy API với code mới")
            return False
    except Exception as e:
        print(f"❌ Không thể kết nối API: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 UPLOAD PLANTS TO API")
    print("=" * 50)
    
    # Nhập URL API
    api_url = input("Nhập URL API Railway (hoặc Enter để dùng localhost): ").strip()
    if api_url:
        API_URL = api_url.rstrip('/')
    else:
        API_URL = "http://localhost:8000"
    
    print(f"🔗 API URL: {API_URL}")
    
    # Test API
    if test_api():
        # Xác nhận upload
        confirm = input("\n🤔 Bạn có muốn upload 100 cây trồng? (y/N): ").strip().lower()
        if confirm in ['y', 'yes']:
            upload_plants()
        else:
            print("❌ Hủy upload")
    else:
        print("❌ Không thể kết nối API. Kiểm tra lại URL.")