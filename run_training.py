#!/usr/bin/env python3
"""
Script chạy training với các options khác nhau
"""

import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description='🌱 Plant Disease Detection Training')
    parser.add_argument('--style', choices=['basic', 'kaggle'], default='kaggle',
                       help='Training style (basic hoặc kaggle)')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Số epochs (default: 20)')
    parser.add_argument('--data', type=str, default='data/PlantVillage',
                       help='Đường dẫn dataset')
    
    args = parser.parse_args()
    
    print("🌱 PLANT DISEASE DETECTION TRAINING")
    print("="*50)
    print(f"Style: {args.style}")
    print(f"Epochs: {args.epochs}")
    print(f"Data: {args.data}")
    print("="*50)
    
    # Kiểm tra dataset
    if not os.path.exists(args.data):
        print(f"❌ Dataset không tồn tại: {args.data}")
        print("Vui lòng tải PlantVillage dataset và đặt vào thư mục data/")
        sys.exit(1)
    
    # Chạy training theo style
    if args.style == 'kaggle':
        from train_kaggle_style import train_kaggle_style
        model, hist1, hist2 = train_kaggle_style(args.data, args.epochs)
        print("✅ Kaggle-style training hoàn thành!")
        
    else:  # basic
        from train import train_model
        model, history = train_model(args.data, args.epochs)
        print("✅ Basic training hoàn thành!")
    
    print(f"🎯 Model đã được lưu!")

if __name__ == "__main__":
    main()