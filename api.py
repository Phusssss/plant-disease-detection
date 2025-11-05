from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests
from PIL import Image
import io
import base64

app = FastAPI(title="Plant Disease Detection API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def call_plant_api(image_bytes):
    """API chẩn đoán bệnh cây trồng tổng quát"""
    url = "https://detect.roboflow.com/plantvillage-dataset/1"
    params = {"api_key": "y0YKSebPyue0doYszJEU"}
    
    response = requests.post(url, params=params, files={"file": image_bytes})
    return response.json()

def call_rice_api(image_bytes):
    """API chẩn đoán bệnh lúa"""
    url = "https://detect.roboflow.com/rice-diseases-qzjka/3"
    params = {"api_key": "y0YKSebPyue0doYszJEU"}
    
    response = requests.post(url, params=params, files={"file": image_bytes})
    return response.json()

def format_rice_disease(class_name):
    """Format tên bệnh lúa"""
    disease_map = {
        'bacterial leaf blight or bacterial blight disease': 'Bệnh cháy lá do vi khuẩn',
        'bacterial leaf streak disease': 'Bệnh vằn lá do vi khuẩn',
        'brown spot disease': 'Bệnh đốm nâu',
        'dirty panicle disease': 'Bệnh bông bẩn',
        'grassy stunt disease': 'Bệnh lùn cỏ',
        'narrow brown spot disease': 'Bệnh đốm nâu hẹp',
        'ragged stunt disease': 'Bệnh lùn rách',
        'rice blast disease': 'Bệnh đạo ôn',
        'sheath blight disease': 'Bệnh khô vỏ lá',
        'tungro disease or yellow orange leaf disease': 'Bệnh tungro (lá vàng cam)'
    }
    return disease_map.get(class_name.lower(), class_name)

@app.get("/")
async def root():
    return {"message": "Plant Disease Detection API", "version": "1.0.0"}

@app.post("/predict/plant")
async def predict_plant_disease(file: UploadFile = File(...)):
    """Chẩn đoán bệnh cây trồng tổng quát"""
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image
        image_bytes = await file.read()
        
        # Call API
        result = call_plant_api(image_bytes)
        
        if 'predictions' in result and result['predictions']:
            pred = result['predictions'][0]
            return {
                "success": True,
                "disease": pred['class'],
                "confidence": pred['confidence'],
                "type": "plant",
                "raw_result": result
            }
        else:
            return {
                "success": False,
                "message": "No disease detected",
                "raw_result": result
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/rice")
async def predict_rice_disease(file: UploadFile = File(...)):
    """Chẩn đoán bệnh lúa chuyên dụng"""
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image
        image_bytes = await file.read()
        
        # Call API
        result = call_rice_api(image_bytes)
        
        if 'predictions' in result and result['predictions']:
            best_pred = max(result['predictions'], key=lambda x: x['confidence'])
            
            return {
                "success": True,
                "disease": best_pred['class'],
                "disease_vietnamese": format_rice_disease(best_pred['class']),
                "confidence": best_pred['confidence'],
                "type": "rice",
                "all_predictions": [
                    {
                        "disease": pred['class'],
                        "disease_vietnamese": format_rice_disease(pred['class']),
                        "confidence": pred['confidence']
                    }
                    for pred in sorted(result['predictions'], key=lambda x: x['confidence'], reverse=True)
                ],
                "raw_result": result
            }
        else:
            return {
                "success": False,
                "message": "No rice disease detected",
                "raw_result": result
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)