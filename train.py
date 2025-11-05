import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import PlantDiseaseModel
import os

def train_model(data_dir, epochs=30, batch_size=16):  # Giảm batch size, tăng epochs
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
    
    # Tính class weights để cân bằng
    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np
    
    class_indices = train_generator.class_indices
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(train_generator.classes),
        y=train_generator.classes
    )
    class_weight_dict = dict(enumerate(class_weights))
    
    print(f"Class weights: {class_weight_dict}")
    
    # Callbacks cải tiến
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, monitor='val_accuracy'),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-7),
        tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_accuracy')
    ]
    
    # Train model với class weights
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
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