import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import numpy as np

class KaggleInspiredPlantModel:
    """Model theo phong cách Kaggle với ResNet50 + FPN + Attention"""
    
    def __init__(self, num_classes=38, input_shape=(224, 224, 3)):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = None
    
    def attention_block(self, x, filters):
        """Attention mechanism như trong Kaggle"""
        # Channel attention
        gap = GlobalAveragePooling2D()(x)
        channel_att = Dense(filters//8, activation='relu')(gap)
        channel_att = Dense(filters, activation='sigmoid')(channel_att)
        channel_att = Reshape((1, 1, filters))(channel_att)
        x = Multiply()([x, channel_att])
        
        # Spatial attention
        spatial_att = Conv2D(1, 7, padding='same', activation='sigmoid')(x)
        x = Multiply()([x, spatial_att])
        
        return x
    
    def fpn_block(self, features):
        """Feature Pyramid Network như Kaggle"""
        # Top-down pathway
        p5 = Conv2D(256, 1, activation='relu')(features[-1])
        p4 = Add()([
            Conv2D(256, 1, activation='relu')(features[-2]),
            UpSampling2D(2)(p5)
        ])
        p3 = Add()([
            Conv2D(256, 1, activation='relu')(features[-3]),
            UpSampling2D(2)(p4)
        ])
        
        # Final feature maps
        p5 = Conv2D(256, 3, padding='same', activation='relu')(p5)
        p4 = Conv2D(256, 3, padding='same', activation='relu')(p4)
        p3 = Conv2D(256, 3, padding='same', activation='relu')(p3)
        
        return [p3, p4, p5]
    
    def create_model(self):
        """Tạo model theo style Kaggle"""
        
        # Input
        inputs = Input(shape=self.input_shape)
        
        # Backbone: ResNet50
        backbone = ResNet50(
            weights='imagenet',
            include_top=False,
            input_tensor=inputs
        )
        
        # Extract features từ multiple layers
        layer_names = ['conv3_block4_out', 'conv4_block6_out', 'conv5_block3_out']
        features = [backbone.get_layer(name).output for name in layer_names]
        
        # Feature Pyramid Network
        fpn_features = self.fpn_block(features)
        
        # Multi-scale feature fusion
        pooled_features = []
        for feat in fpn_features:
            # Attention
            feat = self.attention_block(feat, 256)
            # Global pooling
            pooled = GlobalAveragePooling2D()(feat)
            pooled_features.append(pooled)
        
        # Concatenate multi-scale features
        x = Concatenate()(pooled_features)
        
        # Classification head như Kaggle
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        # Output
        outputs = Dense(self.num_classes, activation='softmax')(x)
        
        # Create model
        model = Model(inputs, outputs, name='KaggleInspiredPlantModel')
        
        # Compile với advanced optimizer
        model.compile(
            optimizer=tf.keras.optimizers.AdamW(
                learning_rate=0.001,
                weight_decay=0.0001
            ),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def unfreeze_for_finetuning(self):
        """Unfreeze model cho fine-tuning như Kaggle"""
        if self.model is None:
            raise ValueError("Model chưa được tạo!")
        
        # Unfreeze backbone
        for layer in self.model.layers:
            if 'resnet50' in layer.name.lower():
                layer.trainable = True
        
        # Compile lại với learning rate thấp hơn
        self.model.compile(
            optimizer=tf.keras.optimizers.AdamW(
                learning_rate=0.0001,  # Thấp hơn cho fine-tuning
                weight_decay=0.0001
            ),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("✅ Model unfrozen for fine-tuning")
    
    def predict_with_tta(self, image, tta_steps=5):
        """Test Time Augmentation như Kaggle"""
        if self.model is None:
            raise ValueError("Model chưa được load!")
        
        predictions = []
        
        for _ in range(tta_steps):
            # Random augmentation
            aug_image = tf.image.random_flip_left_right(image)
            aug_image = tf.image.random_brightness(aug_image, 0.1)
            
            pred = self.model.predict(aug_image, verbose=0)
            predictions.append(pred)
        
        # Average predictions
        final_pred = np.mean(predictions, axis=0)
        return final_pred