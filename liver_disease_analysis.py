# -*- coding: utf-8 -*-
"""
Karaciğer Hastalığı Tahmini ve SHAP Analizi
Liver Disease Prediction with SHAP Analysis

Bu proje karaciğer hastalığı veri setini kullanarak makine öğrenmesi modeli
geliştirir ve SHAP ile model açıklanabilirliği sağlar.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
import shap
import warnings
warnings.filterwarnings('ignore')

# Türkçe karakter desteği için matplotlib ayarları
plt.rcParams['font.family'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class LiverDiseasePredictor:
    def __init__(self):
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        self.best_model = None
        self.explainer = None
        
    def load_data(self, file_path=None):
        """
        Veri setini yükler. Eğer dosya yolu verilmezse örnek veri oluşturur.
        """
        if file_path:
            try:
                self.data = pd.read_csv(file_path)
                print(f"Veri seti başarıyla yüklendi: {self.data.shape}")
            except FileNotFoundError:
                print("Dosya bulunamadı, örnek veri oluşturuluyor...")
                self._create_sample_data()
        else:
            print("Örnek karaciğer hastalığı veri seti oluşturuluyor...")
            self._create_sample_data()
            
        return self.data
    
    def _create_sample_data(self):
        """
        Örnek karaciğer hastalığı veri seti oluşturur
        """
        np.random.seed(42)
        n_samples = 1000
        
        # Karaciğer hastalığı ile ilgili özellikler
        data = {
            'Age': np.random.randint(20, 80, n_samples),
            'Gender': np.random.choice(['Male', 'Female'], n_samples),
            'Total_Bilirubin': np.random.exponential(1.2, n_samples),
            'Direct_Bilirubin': np.random.exponential(0.3, n_samples),
            'Alkaline_Phosphotase': np.random.normal(290, 100, n_samples),
            'Alamine_Aminotransferase': np.random.normal(30, 20, n_samples),
            'Aspartate_Aminotransferase': np.random.normal(35, 25, n_samples),
            'Total_Proteins': np.random.normal(6.5, 1.0, n_samples),
            'Albumin': np.random.normal(3.5, 0.5, n_samples),
            'Albumin_and_Globulin_Ratio': np.random.normal(1.0, 0.3, n_samples)
        }
        
        self.data = pd.DataFrame(data)
        
        # Hastalık durumu için basit bir kural oluştur
        # Yüksek bilirubin, düşük albumin ve yüksek enzim değerleri hastalık riskini artırır
        disease_risk = (
            (self.data['Total_Bilirubin'] > 1.5) * 0.3 +
            (self.data['Direct_Bilirubin'] > 0.5) * 0.2 +
            (self.data['Alkaline_Phosphotase'] > 400) * 0.2 +
            (self.data['Alamine_Aminotransferase'] > 50) * 0.15 +
            (self.data['Aspartate_Aminotransferase'] > 60) * 0.15 +
            (self.data['Albumin'] < 3.0) * 0.2 +
            (self.data['Age'] > 60) * 0.1 +
            np.random.normal(0, 0.1, n_samples)
        )
        
        self.data['Disease'] = (disease_risk > 0.5).astype(int)
        
        print(f"Örnek veri seti oluşturuldu: {self.data.shape}")
        print(f"Hastalık oranı: {self.data['Disease'].mean():.2%}")
    
    def explore_data(self):
        """
        Veri setini keşfeder ve görselleştirir
        """
        print("=== VERİ SETİ BİLGİLERİ ===")
        print(f"Toplam örnek sayısı: {len(self.data)}")
        print(f"Özellik sayısı: {self.data.shape[1] - 1}")
        print(f"Hastalık oranı: {self.data['Disease'].mean():.2%}")
        
        print("\n=== VERİ SETİ İLK 5 SATIR ===")
        print(self.data.head())
        
        print("\n=== VERİ SETİ İSTATİSTİKLERİ ===")
        print(self.data.describe())
        
        print("\n=== EKSİK DEĞERLER ===")
        print(self.data.isnull().sum())
        
        # Görselleştirmeler
        self._create_visualizations()
    
    def _create_visualizations(self):
        """
        Veri görselleştirmeleri oluşturur
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Karaciğer Hastalığı Veri Seti Analizi', fontsize=16, fontweight='bold')
        
        # Hastalık dağılımı
        axes[0, 0].pie(self.data['Disease'].value_counts(), 
                       labels=['Sağlıklı', 'Hastalıklı'], 
                       autopct='%1.1f%%', 
                       colors=['lightgreen', 'lightcoral'])
        axes[0, 0].set_title('Hastalık Dağılımı')
        
        # Yaş dağılımı
        axes[0, 1].hist(self.data['Age'], bins=30, alpha=0.7, color='skyblue')
        axes[0, 1].set_title('Yaş Dağılımı')
        axes[0, 1].set_xlabel('Yaş')
        axes[0, 1].set_ylabel('Frekans')
        
        # Cinsiyet dağılımı
        gender_counts = self.data['Gender'].value_counts()
        axes[0, 2].bar(gender_counts.index, gender_counts.values, color=['lightblue', 'lightpink'])
        axes[0, 2].set_title('Cinsiyet Dağılımı')
        axes[0, 2].set_ylabel('Sayı')
        
        # Bilirubin dağılımı
        axes[1, 0].hist(self.data['Total_Bilirubin'], bins=30, alpha=0.7, color='orange')
        axes[1, 0].set_title('Total Bilirubin Dağılımı')
        axes[1, 0].set_xlabel('Total Bilirubin')
        axes[1, 0].set_ylabel('Frekans')
        
        # Albumin dağılımı
        axes[1, 1].hist(self.data['Albumin'], bins=30, alpha=0.7, color='green')
        axes[1, 1].set_title('Albumin Dağılımı')
        axes[1, 1].set_xlabel('Albumin')
        axes[1, 1].set_ylabel('Frekans')
        
        # Hastalık durumuna göre yaş dağılımı
        healthy_ages = self.data[self.data['Disease'] == 0]['Age']
        diseased_ages = self.data[self.data['Disease'] == 1]['Age']
        
        axes[1, 2].hist([healthy_ages, diseased_ages], 
                       bins=20, alpha=0.7, 
                       label=['Sağlıklı', 'Hastalıklı'],
                       color=['lightgreen', 'lightcoral'])
        axes[1, 2].set_title('Hastalık Durumuna Göre Yaş Dağılımı')
        axes[1, 2].set_xlabel('Yaş')
        axes[1, 2].set_ylabel('Frekans')
        axes[1, 2].legend()
        
        plt.tight_layout()
        plt.savefig('results/data_exploration.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Korelasyon matrisi
        plt.figure(figsize=(12, 10))
        correlation_matrix = self.data.select_dtypes(include=[np.number]).corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Özellikler Arası Korelasyon Matrisi', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('results/correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def preprocess_data(self):
        """
        Veriyi ön işleme tabi tutar
        """
        print("=== VERİ ÖN İŞLEME ===")
        
        # Kopya oluştur
        df = self.data.copy()
        
        # Kategorik değişkenleri encode et
        le = LabelEncoder()
        df['Gender'] = le.fit_transform(df['Gender'])
        
        # Hedef değişkeni ayır
        X = df.drop('Disease', axis=1)
        y = df['Disease']
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Özellikleri ölçeklendir
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Eğitim seti boyutu: {self.X_train.shape}")
        print(f"Test seti boyutu: {self.X_test.shape}")
        print(f"Eğitim seti hastalık oranı: {self.y_train.mean():.2%}")
        print(f"Test seti hastalık oranı: {self.y_test.mean():.2%}")
        
        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test
    
    def train_models(self):
        """
        Farklı ML modellerini eğitir
        """
        print("=== MODEL EĞİTİMİ ===")
        
        # Model tanımları
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'SVM': SVC(random_state=42, probability=True)
        }
        
        # Her modeli eğit ve değerlendir
        for name, model in models.items():
            print(f"\n{name} eğitiliyor...")
            
            # Modeli eğit
            model.fit(self.X_train_scaled, self.y_train)
            
            # Tahminler
            y_pred = model.predict(self.X_test_scaled)
            y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
            
            # Metrikler
            accuracy = accuracy_score(self.y_test, y_pred)
            auc_score = roc_auc_score(self.y_test, y_pred_proba)
            
            # Cross-validation
            cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=5)
            
            print(f"Accuracy: {accuracy:.4f}")
            print(f"AUC Score: {auc_score:.4f}")
            print(f"CV Score (mean): {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            # Modeli kaydet
            self.models[name] = {
                'model': model,
                'accuracy': accuracy,
                'auc': auc_score,
                'cv_scores': cv_scores,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
        
        # En iyi modeli seç
        best_model_name = max(self.models.keys(), key=lambda x: self.models[x]['auc'])
        self.best_model = self.models[best_model_name]['model']
        
        print(f"\nEn iyi model: {best_model_name}")
        print(f"En iyi AUC skoru: {self.models[best_model_name]['auc']:.4f}")
        
        return self.models
    
    def evaluate_models(self):
        """
        Modelleri detaylı olarak değerlendirir
        """
        print("=== MODEL DEĞERLENDİRME ===")
        
        # En iyi model için detaylı rapor
        best_model_name = max(self.models.keys(), key=lambda x: self.models[x]['auc'])
        best_model_info = self.models[best_model_name]
        
        print(f"\n{best_model_name} - Detaylı Performans:")
        print(classification_report(self.y_test, best_model_info['predictions']))
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_test, best_model_info['predictions'])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Sağlıklı', 'Hastalıklı'],
                   yticklabels=['Sağlıklı', 'Hastalıklı'])
        plt.title(f'{best_model_name} - Confusion Matrix')
        plt.ylabel('Gerçek Değerler')
        plt.xlabel('Tahmin Edilen Değerler')
        plt.tight_layout()
        plt.savefig('results/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # ROC Curve
        plt.figure(figsize=(10, 8))
        
        for name, model_info in self.models.items():
            fpr, tpr, _ = roc_curve(self.y_test, model_info['probabilities'])
            plt.plot(fpr, tpr, label=f'{name} (AUC = {model_info["auc"]:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Model Karşılaştırması')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('results/roc_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Model performans karşılaştırması
        model_names = list(self.models.keys())
        accuracies = [self.models[name]['accuracy'] for name in model_names]
        auc_scores = [self.models[name]['auc'] for name in model_names]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy karşılaştırması
        bars1 = ax1.bar(model_names, accuracies, color='skyblue', alpha=0.7)
        ax1.set_title('Model Accuracy Karşılaştırması')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='x', rotation=45)
        
        # Accuracy değerlerini bar üzerine yaz
        for bar, acc in zip(bars1, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        # AUC karşılaştırması
        bars2 = ax2.bar(model_names, auc_scores, color='lightcoral', alpha=0.7)
        ax2.set_title('Model AUC Score Karşılaştırması')
        ax2.set_ylabel('AUC Score')
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis='x', rotation=45)
        
        # AUC değerlerini bar üzerine yaz
        for bar, auc in zip(bars2, auc_scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{auc:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('results/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def shap_analysis(self):
        """
        SHAP analizi ile model açıklanabilirliği
        """
        print("=== SHAP ANALİZİ ===")
        
        if self.best_model is None:
            print("Önce modelleri eğitin!")
            return
        
        # SHAP explainer oluştur
        self.explainer = shap.Explainer(self.best_model, self.X_train_scaled)
        shap_values = self.explainer(self.X_test_scaled)
        
        # Feature importance
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, self.X_test_scaled, 
                         feature_names=self.X_train.columns,
                         show=False)
        plt.title('SHAP Feature Importance', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('results/shap_summary.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Feature importance bar plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, self.X_test_scaled, 
                         feature_names=self.X_train.columns,
                         plot_type="bar", show=False)
        plt.title('SHAP Feature Importance (Bar Plot)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('results/shap_bar.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Waterfall plot (ilk 5 örnek için)
        for i in range(min(5, len(self.X_test_scaled))):
            plt.figure(figsize=(12, 8))
            shap.waterfall_plot(shap_values[i], show=False)
            plt.title(f'SHAP Waterfall Plot - Örnek {i+1}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'results/shap_waterfall_{i+1}.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # Partial dependence plot
        plt.figure(figsize=(15, 10))
        shap.partial_dependence_plot(
            "Total_Bilirubin", self.best_model.predict, self.X_test_scaled,
            ice=False, model_expected_value=True, feature_expected_value=True,
            show=False
        )
        plt.title('Partial Dependence Plot - Total Bilirubin', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('results/shap_pdp_bilirubin.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("SHAP analizi tamamlandı. Görseller 'results' klasörüne kaydedildi.")
        
        return shap_values
    
    def predict_new_sample(self, sample_data):
        """
        Yeni bir örnek için tahmin yapar
        """
        if self.best_model is None:
            print("Önce modelleri eğitin!")
            return None
        
        # Veriyi ölçeklendir
        sample_scaled = self.scaler.transform([sample_data])
        
        # Tahmin
        prediction = self.best_model.predict(sample_scaled)[0]
        probability = self.best_model.predict_proba(sample_scaled)[0]
        
        print(f"Tahmin: {'Hastalıklı' if prediction == 1 else 'Sağlıklı'}")
        print(f"Hastalık olasılığı: {probability[1]:.2%}")
        
        return prediction, probability
    
    def run_complete_analysis(self):
        """
        Tam analizi çalıştırır
        """
        print("=== KARACİĞER HASTALIĞI TAHMİN ANALİZİ ===")
        
        # 1. Veri yükleme
        self.load_data()
        
        # 2. Veri keşfi
        self.explore_data()
        
        # 3. Veri ön işleme
        self.preprocess_data()
        
        # 4. Model eğitimi
        self.train_models()
        
        # 5. Model değerlendirme
        self.evaluate_models()
        
        # 6. SHAP analizi
        self.shap_analysis()
        
        print("\n=== ANALİZ TAMAMLANDI ===")
        print("Tüm sonuçlar 'results' klasörüne kaydedildi.")


def main():
    """
    Ana fonksiyon
    """
    # Analiz sınıfını oluştur
    analyzer = LiverDiseasePredictor()
    
    # Tam analizi çalıştır
    analyzer.run_complete_analysis()
    
    # Örnek tahmin
    print("\n=== ÖRNEK TAHMİN ===")
    sample = [45, 1, 2.5, 1.2, 350, 45, 55, 6.0, 3.2, 1.1]  # Örnek hasta verisi
    analyzer.predict_new_sample(sample)


if __name__ == "__main__":
    main()
