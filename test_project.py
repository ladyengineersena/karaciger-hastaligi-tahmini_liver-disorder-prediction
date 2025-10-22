# -*- coding: utf-8 -*-
"""
Test Script - Karaciğer Hastalığı Tahmini
Test Script - Liver Disease Prediction

Bu script projenin doğru çalışıp çalışmadığını test eder.
"""

import sys
import os

def test_imports():
    """Gerekli kütüphanelerin import edilip edilemediğini test eder"""
    try:
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        import shap
        print("✅ Tüm kütüphaneler başarıyla import edildi!")
        return True
    except ImportError as e:
        print(f"❌ Kütüphane import hatası: {e}")
        return False

def test_main_script():
    """Ana script'in çalışıp çalışmadığını test eder"""
    try:
        from liver_disease_analysis import LiverDiseasePredictor
        print("✅ Ana script başarıyla import edildi!")
        
        # Analiz sınıfını oluştur
        analyzer = LiverDiseasePredictor()
        print("✅ Analiz sınıfı başarıyla oluşturuldu!")
        
        # Veri yükleme testi
        data = analyzer.load_data()
        print(f"✅ Veri seti başarıyla yüklendi: {data.shape}")
        
        # Veri ön işleme testi
        X_train_scaled, X_test_scaled, y_train, y_test = analyzer.preprocess_data()
        print("✅ Veri ön işleme başarıyla tamamlandı!")
        
        # Model eğitimi testi
        models = analyzer.train_models()
        print("✅ Model eğitimi başarıyla tamamlandı!")
        
        return True
    except Exception as e:
        print(f"❌ Ana script test hatası: {e}")
        return False

def test_directories():
    """Gerekli klasörlerin var olup olmadığını test eder"""
    required_dirs = ['data', 'models', 'notebooks', 'results']
    
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✅ {dir_name} klasörü mevcut!")
        else:
            print(f"❌ {dir_name} klasörü bulunamadı!")
            return False
    
    return True

def main():
    """Ana test fonksiyonu"""
    print("=== KARACİĞER HASTALIĞI TAHMİN PROJESİ TEST ===")
    print()
    
    # Test sonuçları
    tests = [
        ("Klasör Testi", test_directories),
        ("Kütüphane Import Testi", test_imports),
        ("Ana Script Testi", test_main_script)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"--- {test_name} ---")
        if test_func():
            passed += 1
        print()
    
    print("=== TEST SONUÇLARI ===")
    print(f"Geçen testler: {passed}/{total}")
    
    if passed == total:
        print("🎉 Tüm testler başarıyla geçildi! Proje hazır.")
    else:
        print("⚠️ Bazı testler başarısız oldu. Lütfen hataları kontrol edin.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
