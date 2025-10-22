# Karaciğer Hastalığı Tahmini ve SHAP Analizi
# Liver Disease Prediction with SHAP Analysis

Bu proje, karaciğer hastalığı veri setini kullanarak makine öğrenmesi modelleri geliştirir ve SHAP (SHapley Additive exPlanations) ile model açıklanabilirliği sağlar.

## 🎯 Proje Amacı

Karaciğer hastalığı erken teşhisi için makine öğrenmesi modelleri geliştirmek ve bu modellerin kararlarını SHAP analizi ile açıklamak.

## 📊 Veri Seti

Proje, aşağıdaki özellikleri içeren karaciğer hastalığı veri setini kullanır:

- **Age**: Yaş
- **Gender**: Cinsiyet (Male/Female)
- **Total_Bilirubin**: Total bilirubin seviyesi
- **Direct_Bilirubin**: Direkt bilirubin seviyesi
- **Alkaline_Phosphotase**: Alkalin fosfataz seviyesi
- **Alamine_Aminotransferase**: ALT seviyesi
- **Aspartate_Aminotransferase**: AST seviyesi
- **Total_Proteins**: Total protein seviyesi
- **Albumin**: Albumin seviyesi
- **Albumin_and_Globulin_Ratio**: Albumin/Globulin oranı
- **Disease**: Hastalık durumu (0: Sağlıklı, 1: Hastalıklı)

## 🚀 Kurulum

### Gereksinimler

- Python 3.7+
- Gerekli kütüphaneler (requirements.txt dosyasında listelenmiştir)

### Kurulum Adımları

1. Projeyi klonlayın:
```bash
git clone https://github.com/kullaniciadi/liver-disease-prediction.git
cd liver-disease-prediction
```

2. Sanal ortam oluşturun (önerilen):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# veya
venv\Scripts\activate  # Windows
```

3. Gerekli kütüphaneleri yükleyin:
```bash
pip install -r requirements.txt
```

## 📁 Proje Yapısı

```
liver-disease-prediction/
├── data/                   # Veri dosyaları
├── models/                 # Eğitilmiş modeller
├── notebooks/              # Jupyter notebook'lar
│   └── liver_disease_analysis.ipynb
├── results/                # Analiz sonuçları ve görseller
├── liver_disease_analysis.py  # Ana analiz kodu
├── requirements.txt        # Gerekli kütüphaneler
└── README.md              # Bu dosya
```

## 🔧 Kullanım

### Python Script ile Çalıştırma

```bash
python liver_disease_analysis.py
```

### Jupyter Notebook ile Çalıştırma

```bash
jupyter notebook notebooks/liver_disease_analysis.ipynb
```

## 📈 Analiz Süreci

1. **Veri Yükleme ve Keşif**: Veri setinin yüklenmesi ve temel istatistiklerin incelenmesi
2. **Veri Ön İşleme**: Eksik değerlerin işlenmesi ve özelliklerin ölçeklendirilmesi
3. **Model Eğitimi**: Farklı ML algoritmalarının eğitilmesi
4. **Model Değerlendirme**: Performans metriklerinin hesaplanması
5. **SHAP Analizi**: Model açıklanabilirliğinin sağlanması

## 🤖 Kullanılan Modeller

- **Logistic Regression**: Temel sınıflandırma modeli
- **Random Forest**: Ensemble öğrenme modeli
- **Gradient Boosting**: Gelişmiş ensemble modeli
- **Support Vector Machine (SVM)**: Kernel tabanlı model

## 📊 SHAP Analizi

SHAP analizi ile:
- Özellik önemliliği belirlenir
- Her tahmin için açıklama sağlanır
- Model kararlarının şeffaflığı artırılır
- Klinik karar verme sürecine destek olunur

## 📋 Sonuçlar

Analiz sonuçları `results/` klasöründe saklanır:
- Veri keşif görselleri
- Model performans grafikleri
- SHAP analiz görselleri
- Confusion matrix
- ROC eğrileri

## 🏥 Klinik Öneriler

Model analizi sonucunda önemli risk faktörleri:
- Yüksek bilirubin değerleri
- Düşük albumin seviyeleri
- Yüksek karaciğer enzimleri (ALT, AST)
- İleri yaş
- Cinsiyet faktörü

## 🔍 Örnek Kullanım

```python
from liver_disease_analysis import LiverDiseasePredictor

# Analiz sınıfını oluştur
analyzer = LiverDiseasePredictor()

# Veri setini yükle
analyzer.load_data()

# Tam analizi çalıştır
analyzer.run_complete_analysis()

# Yeni hasta için tahmin
sample_data = [45, 1, 2.5, 1.2, 350, 45, 55, 6.0, 3.2, 1.1]
prediction, probability = analyzer.predict_new_sample(sample_data)
```

## 📝 Notlar

- Proje Türkçe karakter desteği ile geliştirilmiştir
- Tüm görseller ve çıktılar Türkçe etiketler içerir
- Model performansı gerçek veri setine göre değişebilir
- Klinik kullanım için doktor onayı gereklidir

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/AmazingFeature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add some AmazingFeature'`)
4. Branch'inizi push edin (`git push origin feature/AmazingFeature`)
5. Pull Request oluşturun

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakın.

## 👥 Yazarlar

- Proje Geliştirici - [GitHub Profili](https://github.com/kullaniciadi)

## 🙏 Teşekkürler

- SHAP kütüphanesi geliştiricilerine
- Scikit-learn ekibine
- Açık kaynak topluluğuna

## 📞 İletişim

Sorularınız için: [email@example.com](mailto:email@example.com)

Proje Linki: [https://github.com/kullaniciadi/liver-disease-prediction](https://github.com/kullaniciadi/liver-disease-prediction)
