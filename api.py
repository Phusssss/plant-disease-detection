from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
from PIL import Image
import io
import sqlite3
import os
from datetime import datetime
from typing import List, Optional

app = FastAPI(title="Plant Disease Detection API with Database", version="1.0.0")

# Pydantic models
class PlantCreate(BaseModel):
    name: str
    scientific_name: Optional[str] = None
    description: Optional[str] = None
    care_instructions: Optional[str] = None

class PlantResponse(BaseModel):
    id: int
    name: str
    scientific_name: Optional[str]
    description: Optional[str]
    care_instructions: Optional[str]
    created_at: str

class DiagnosisResponse(BaseModel):
    id: int
    plant_name: Optional[str]
    disease: str
    confidence: float
    timestamp: str

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database setup
def init_db():
    conn = sqlite3.connect('plants.db')
    cursor = conn.cursor()
    
    # Plants table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS plants (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            scientific_name TEXT,
            description TEXT,
            care_instructions TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Diagnoses table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS diagnoses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plant_id INTEGER,
            disease TEXT,
            confidence REAL,
            type TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (plant_id) REFERENCES plants (id)
        )
    ''')
    
    # Insert default plants
    default_plants = [
        ('Táo', 'Malus domestica', 'Cây ăn quả phổ biến', 'Tưới nước đều đặn, cần ánh sáng'),
        ('Cà chua', 'Solanum lycopersicum', 'Cây rau quả dễ trồng', 'Cần nhiều nước và ánh sáng'),
        ('Lúa', 'Oryza sativa', 'Cây lương thực chính', 'Trồng trong môi trường ẩm ướt'),
        ('Ngô', 'Zea mays', 'Cây ngũ cốc quan trọng', 'Cần đất tơi xốp và phân bón'),
        ('Khoai tây', 'Solanum tuberosum', 'Cây củ dinh dưỡng', 'Trồng trong đất thoát nước tốt')
    ]
    
    cursor.executemany('''
        INSERT OR IGNORE INTO plants (name, scientific_name, description, care_instructions)
        VALUES (?, ?, ?, ?)
    ''', default_plants)
    
    conn.commit()
    conn.close()

init_db()

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
    return {"message": "Plant Disease Detection API with Database", "version": "1.0.0"}

# Plants CRUD endpoints
@app.get("/plants", response_model=List[PlantResponse])
async def get_plants():
    """Lấy danh sách tất cả cây trồng"""
    conn = sqlite3.connect('plants.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM plants ORDER BY name')
    plants = cursor.fetchall()
    conn.close()
    
    return [
        PlantResponse(
            id=plant[0],
            name=plant[1],
            scientific_name=plant[2],
            description=plant[3],
            care_instructions=plant[4],
            created_at=plant[5]
        )
        for plant in plants
    ]

@app.post("/plants", response_model=PlantResponse)
async def create_plant(plant: PlantCreate):
    """Thêm cây trồng mới"""
    conn = sqlite3.connect('plants.db')
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT INTO plants (name, scientific_name, description, care_instructions)
            VALUES (?, ?, ?, ?)
        ''', (plant.name, plant.scientific_name, plant.description, plant.care_instructions))
        
        plant_id = cursor.lastrowid
        conn.commit()
        
        # Get created plant
        cursor.execute('SELECT * FROM plants WHERE id = ?', (plant_id,))
        created_plant = cursor.fetchone()
        conn.close()
        
        return PlantResponse(
            id=created_plant[0],
            name=created_plant[1],
            scientific_name=created_plant[2],
            description=created_plant[3],
            care_instructions=created_plant[4],
            created_at=created_plant[5]
        )
        
    except sqlite3.IntegrityError:
        conn.close()
        raise HTTPException(status_code=400, detail="Plant name already exists")

@app.get("/plants/{plant_id}", response_model=PlantResponse)
async def get_plant(plant_id: int):
    """Lấy thông tin cây trồng theo ID"""
    conn = sqlite3.connect('plants.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM plants WHERE id = ?', (plant_id,))
    plant = cursor.fetchone()
    conn.close()
    
    if not plant:
        raise HTTPException(status_code=404, detail="Plant not found")
    
    return PlantResponse(
        id=plant[0],
        name=plant[1],
        scientific_name=plant[2],
        description=plant[3],
        care_instructions=plant[4],
        created_at=plant[5]
    )

@app.delete("/plants/{plant_id}")
async def delete_plant(plant_id: int):
    """Xóa cây trồng"""
    conn = sqlite3.connect('plants.db')
    cursor = conn.cursor()
    cursor.execute('DELETE FROM plants WHERE id = ?', (plant_id,))
    
    if cursor.rowcount == 0:
        conn.close()
        raise HTTPException(status_code=404, detail="Plant not found")
    
    conn.commit()
    conn.close()
    return {"message": "Plant deleted successfully"}

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
            
            # Save to database
            conn = sqlite3.connect('plants.db')
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO diagnoses (disease, confidence, type)
                VALUES (?, ?, ?)
            ''', (pred['class'], pred['confidence'], 'plant'))
            diagnosis_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            return {
                "success": True,
                "diagnosis_id": diagnosis_id,
                "disease": pred['class'],
                "confidence": pred['confidence'],
                "type": "plant",
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "success": False,
                "message": "No disease detected"
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
            
            # Save to database
            conn = sqlite3.connect('plants.db')
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO diagnoses (disease, confidence, type)
                VALUES (?, ?, ?)
            ''', (best_pred['class'], best_pred['confidence'], 'rice'))
            diagnosis_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            return {
                "success": True,
                "diagnosis_id": diagnosis_id,
                "disease": best_pred['class'],
                "disease_vietnamese": format_rice_disease(best_pred['class']),
                "confidence": best_pred['confidence'],
                "type": "rice",
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "success": False,
                "message": "No rice disease detected"
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/diagnoses", response_model=List[DiagnosisResponse])
async def get_diagnoses(limit: int = 50):
    """Lấy lịch sử chẩn đoán"""
    conn = sqlite3.connect('plants.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT d.id, p.name, d.disease, d.confidence, d.timestamp
        FROM diagnoses d
        LEFT JOIN plants p ON d.plant_id = p.id
        ORDER BY d.timestamp DESC
        LIMIT ?
    ''', (limit,))
    
    diagnoses = cursor.fetchall()
    conn.close()
    
    return [
        DiagnosisResponse(
            id=diag[0],
            plant_name=diag[1],
            disease=diag[2],
            confidence=diag[3],
            timestamp=diag[4]
        )
        for diag in diagnoses
    ]

@app.get("/stats")
async def get_stats():
    """Thống kê hệ thống"""
    conn = sqlite3.connect('plants.db')
    cursor = conn.cursor()
    
    # Total plants
    cursor.execute('SELECT COUNT(*) FROM plants')
    total_plants = cursor.fetchone()[0]
    
    # Total diagnoses
    cursor.execute('SELECT COUNT(*) FROM diagnoses')
    total_diagnoses = cursor.fetchone()[0]
    
    # Diagnoses by type
    cursor.execute('SELECT type, COUNT(*) FROM diagnoses GROUP BY type')
    by_type = dict(cursor.fetchall())
    
    conn.close()
    
    return {
        "total_plants": total_plants,
        "total_diagnoses": total_diagnoses,
        "diagnoses_by_type": by_type
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)