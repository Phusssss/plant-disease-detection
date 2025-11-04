import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import PlantDiseaseModel
import os

def train_model(data_dir, epochs=20, batch_size=32):
    # Tạo data generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )
    
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )
    
    validation_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )
    
    # Tạo và train model
    plant_model = PlantDiseaseModel(num_classes=len(train_generator.class_indices))
    model = plant_model.create_model()
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=3)
    ]
    
    # Train model
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=callbacks
    )
    
    # Lưu model
    plant_model.save_model('plant_disease_model.h5')
    print("Model đã được lưu thành công!")
    
    return plant_model, history

if __name__ == "__main__":
    # Thay đổi đường dẫn này đến thư mục chứa dữ liệu
    data_directory = "data/PlantVillage"
    
    if os.path.exists(data_directory):
        model, history = train_model(data_directory)
    else:
        print(f"Không tìm thấy thư mục dữ liệu: {data_directory}")
        print("Vui lòng tải dataset PlantVillage và đặt vào thư mục 'data/PlantVillage'")