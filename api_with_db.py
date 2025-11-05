from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests
from PIL import Image
import io
import os
from datetime import datetime
import sqlite3
import json

app = FastAPI(title="Plant Disease Detection API with Database", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database setup
def init_db():
    conn = sqlite3.connect('diagnoses.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS diagnoses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            type TEXT NOT NULL,
            disease TEXT,
            disease_vietnamese TEXT,
            confidence REAL,
            success BOOLEAN,
            raw_result TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def save_diagnosis(type_plant, disease, disease_vn, confidence, success, raw_result):
    conn = sqlite3.connect('diagnoses.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO diagnoses (type, disease, disease_vietnamese, confidence, success, raw_result)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (type_plant, disease, disease_vn, confidence, success, json.dumps(raw_result)))
    diagnosis_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return diagnosis_id

def call_plant_api(image_bytes):
    url = "https://detect.roboflow.com/plantvillage-dataset/1"
    params = {"api_key": "y0YKSebPyue0doYszJEU"}
    response = requests.post(url, params=params, files={"file": image_bytes})
    return response.json()

def call_rice_api(image_bytes):
    url = "https://detect.roboflow.com/rice-diseases-qzjka/3"
    params = {"api_key": "y0YKSebPyue0doYszJEU"}
    response = requests.post(url, params=params, files={"file": image_bytes})
    return response.json()

def format_rice_disease(class_name):
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
    return {"message": "Plant Disease Detection API with Database", "version": "1.0.0"}

@app.post("/predict/plant")
async def predict_plant_disease(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        image_bytes = await file.read()
        result = call_plant_api(image_bytes)
        
        if 'predictions' in result and result['predictions']:
            pred = result['predictions'][0]
            diagnosis_id = save_diagnosis("plant", pred['class'], pred['class'], pred['confidence'], True, result)
            
            return {
                "success": True,
                "diagnosis_id": diagnosis_id,
                "disease": pred['class'],
                "confidence": pred['confidence'],
                "type": "plant",
                "timestamp": datetime.now().isoformat()
            }
        else:
            diagnosis_id = save_diagnosis("plant", None, None, 0, False, result)
            return {
                "success": False,
                "diagnosis_id": diagnosis_id,
                "message": "No disease detected"
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/rice")
async def predict_rice_disease(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        image_bytes = await file.read()
        result = call_rice_api(image_bytes)
        
        if 'predictions' in result and result['predictions']:
            best_pred = max(result['predictions'], key=lambda x: x['confidence'])
            disease_vn = format_rice_disease(best_pred['class'])
            
            diagnosis_id = save_diagnosis("rice", best_pred['class'], disease_vn, best_pred['confidence'], True, result)
            
            return {
                "success": True,
                "diagnosis_id": diagnosis_id,
                "disease": best_pred['class'],
                "disease_vietnamese": disease_vn,
                "confidence": best_pred['confidence'],
                "type": "rice",
                "timestamp": datetime.now().isoformat()
            }
        else:
            diagnosis_id = save_diagnosis("rice", None, None, 0, False, result)
            return {
                "success": False,
                "diagnosis_id": diagnosis_id,
                "message": "No rice disease detected"
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
async def get_diagnosis_history(limit: int = 50):
    """Lấy lịch sử chẩn đoán"""
    conn = sqlite3.connect('diagnoses.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, timestamp, type, disease, disease_vietnamese, confidence, success
        FROM diagnoses 
        ORDER BY timestamp DESC 
        LIMIT ?
    ''', (limit,))
    
    results = cursor.fetchall()
    conn.close()
    
    history = []
    for row in results:
        history.append({
            "id": row[0],
            "timestamp": row[1],
            "type": row[2],
            "disease": row[3],
            "disease_vietnamese": row[4],
            "confidence": row[5],
            "success": bool(row[6])
        })
    
    return {"history": history, "total": len(history)}

@app.get("/stats")
async def get_stats():
    """Thống kê chẩn đoán"""
    conn = sqlite3.connect('diagnoses.db')
    cursor = conn.cursor()
    
    # Tổng số chẩn đoán
    cursor.execute('SELECT COUNT(*) FROM diagnoses')
    total = cursor.fetchone()[0]
    
    # Thành công
    cursor.execute('SELECT COUNT(*) FROM diagnoses WHERE success = 1')
    successful = cursor.fetchone()[0]
    
    # Theo loại
    cursor.execute('SELECT type, COUNT(*) FROM diagnoses GROUP BY type')
    by_type = dict(cursor.fetchall())
    
    # Top bệnh
    cursor.execute('''
        SELECT disease_vietnamese, COUNT(*) as count 
        FROM diagnoses 
        WHERE success = 1 AND disease_vietnamese IS NOT NULL
        GROUP BY disease_vietnamese 
        ORDER BY count DESC 
        LIMIT 10
    ''')
    top_diseases = [{"disease": row[0], "count": row[1]} for row in cursor.fetchall()]
    
    conn.close()
    
    return {
        "total_diagnoses": total,
        "successful_diagnoses": successful,
        "success_rate": successful/total if total > 0 else 0,
        "by_type": by_type,
        "top_diseases": top_diseases
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)