import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model_kaggle_inspired import KaggleInspiredPlantModel
import os
import numpy as np

class FocalLoss(tf.keras.losses.Loss):
    """Focal Loss như trong Kaggle code"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def call(self, y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # Compute focal loss
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        focal_weight = tf.pow(1 - pt, self.gamma)
        
        if self.alpha >= 0:
            alpha_t = tf.where(tf.equal(y_true, 1), self.alpha, 1 - self.alpha)
            focal_weight = alpha_t * focal_weight
        
        ce_loss = -y_true * tf.math.log(y_pred)
        focal_loss = focal_weight * ce_loss
        
        return tf.reduce_sum(focal_loss, axis=1)

def create_advanced_generators(data_dir, batch_size=4):
    """Data generators theo style Kaggle"""
    
    # Advanced augmentation như Kaggle
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.7, 1.3],
        channel_shift_range=0.2,
        fill_mode='nearest',
        validation_split=0.2
    )
    
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    return train_generator, val_generator

def train_kaggle_style(data_dir, epochs=20):
    """Training theo style Kaggle với advanced techniques"""
    
    print("🚀 KAGGLE-INSPIRED TRAINING")
    print("="*50)
    
    # Create generators với batch size nhỏ như Kaggle
    train_gen, val_gen = create_advanced_generators(data_dir, batch_size=4)
    
    print(f"📊 Dataset Info:")
    print(f"  - Training samples: {train_gen.samples}")
    print(f"  - Validation samples: {val_gen.samples}")
    print(f"  - Classes: {len(train_gen.class_indices)}")
    print(f"  - Batch size: 4 (như Kaggle)")
    
    # Compute class weights (balanced như Kaggle)
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(train_gen.classes),
        y=train_gen.classes
    )
    class_weight_dict = dict(enumerate(class_weights))
    
    # Create model
    model = KaggleInspiredPlantModel(num_classes=len(train_gen.class_indices))
    model = model.create_model()
    
    print(f"📊 Model Info:")
    print(f"  - Architecture: ResNet50 + FPN + Attention")
    print(f"  - Total params: {model.count_params():,}")
    print(f"  - Trainable params: {sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]):,}")
    
    # Advanced callbacks như Kaggle
    callbacks = [
        # Early stopping với patience cao
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        
        # ReduceLR như Kaggle
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        
        # Model checkpoint
        tf.keras.callbacks.ModelCheckpoint(
            'kaggle_inspired_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        
        # Learning rate scheduler
        tf.keras.callbacks.LearningRateScheduler(
            lambda epoch: 0.001 * (0.9 ** epoch),
            verbose=1
        )
    ]
    
    # STAGE 1: Feature extraction (như Kaggle)
    print(f"\n🔥 STAGE 1: Feature Extraction ({epochs//2} epochs)")
    print("-" * 40)
    
    history1 = model.fit(
        train_gen,
        epochs=epochs//2,
        validation_data=val_gen,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )
    
    # STAGE 2: Fine-tuning (như Kaggle)
    print(f"\n🔥 STAGE 2: Fine-tuning ({epochs//2} epochs)")
    print("-" * 40)
    
    # Unfreeze model
    plant_model = KaggleInspiredPlantModel()
    plant_model.model = model
    plant_model.unfreeze_for_finetuning()
    
    # Continue training
    history2 = model.fit(
        train_gen,
        epochs=epochs//2,
        validation_data=val_gen,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )
    
    # Final results
    final_val_acc = max(history2.history['val_accuracy'])
    final_train_acc = max(history2.history['accuracy'])
    
    print(f"\n✅ KAGGLE-STYLE TRAINING COMPLETED!")
    print(f"="*50)
    print(f"🎯 Final Results:")
    print(f"  - Training Accuracy: {final_train_acc:.4f}")
    print(f"  - Validation Accuracy: {final_val_acc:.4f}")
    print(f"  - Architecture: ResNet50 + FPN + Attention")
    print(f"  - Expected: 80-90% accuracy (như Kaggle)")
    print(f"  - Model saved: kaggle_inspired_model.h5")
    
    return plant_model, history1, history2

if __name__ == "__main__":
    data_directory = "data/PlantVillage"
    
    if os.path.exists(data_directory):
        model, hist1, hist2 = train_kaggle_style(data_directory, epochs=20)
    else:
        print(f"❌ Data directory not found: {data_directory}")
        print("Please make sure PlantVillage dataset is available")