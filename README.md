# 🏦 Müşteri Kaybı (Churn) Tahmin Modeli
### ING Hubs Türkiye Datathon — End-to-End Machine Learning Project

> **Bir banka müşterisinin yakın gelecekte ayrılıp ayrılmayacağını önceden tahmin edebilir miyiz?**  
> Bu proje, ING Hubs Türkiye Datathon kapsamında müşteri işlem geçmişi ve demografik veriler kullanılarak churn tahmini yapan uçtan uca bir ML pipeline'ı sunmaktadır.

---

## 📌 Genel Bakış

Üç farklı veri kaynağı (`customers.csv`, `customer_history.csv`, `referance_data.csv`) birleştirilerek analitik bir temel tablo (ABT) oluşturulmuş ve müşteri kaybı ikili sınıflandırma problemi olarak modellenmiştir.

Bu proje aynı zamanda **ilk datathon deneyimim** olma özelliğini taşımaktadır. Yarışma sürecinde veri sızıntısı önleme, zaman bazlı özellik mühendisliği ve resmi yarışma metriği üzerinde optimizasyon gibi gerçek dünya problemlerini ilk kez bu ölçekte ele aldım. Retrospektif olarak fark ettiğim bazı iyileştirme alanları (cross-validation eksikliği, final model early stopping stratejisi) `Limitasyonlar` bölümünde dürüstçe belgelenmiştir — bu hataların farkında olmak, bir sonraki projede daha iyi bir başlangıç noktası sağlıyor.

**Öne çıkan metodolojik kararlar:**
- **Veri sızıntısını önlemek** için tüm özellikler `date < ref_date` filtresiyle hesaplandı — model gelecekteki veriyi görmüyor
- **Trend ve oran özellikleri** türetildi: son 3 ay davranışının genel davranışa oranı güçlü churn sinyali taşıyor
- **Encoding ve ölçeklendirme** split'ten sonra, sadece eğitim verisi üzerinden yapıldı
- Yarışmanın **resmi metriği** (Gini + Recall@10% + Lift@10% bileşik skoru) üzerinde **Optuna** ile hiperparametre optimizasyonu uygulandı

---

## 📊 Veri Seti

| Kaynak | İçerik |
|---|---|
| `customers.csv` | Müşteri demografik bilgileri (yaş, cinsiyet, şehir, meslek vb.) |
| `customer_history.csv` | Aylık işlem geçmişi (KK harcaması, EFT sayısı, aktif ürün vb.) |
| `referance_data.csv` | Churn etiketleri ve referans tarihleri |
| `sample_submission.csv` | Tahmin yapılacak müşteri listesi |

---

## 🔁 Pipeline
```
Ham Veri (3 Kaynak)
   └─ Time-Aware Birleştirme    (date < ref_date → sızıntı önleme)
        └─ Özellik Mühendisliği  (genel, trend, oran özellikleri)
             └─ EDA              (hedef değişken, korelasyon, kategorik analiz)
                  └─ Ön İşleme   (gruplama, OHE, StandardScaler)
                       └─ Train/Test Split  (stratified, 80/20)
                            └─ XGBoost + Optuna (resmi metrik üzerinde)
                                 └─ Submission Dosyası
```

---

## 🛠️ Özellik Mühendisliği

| Özellik Grubu | Örnekler | Açıklama |
|---|---|---|
| **Genel Davranış** | `cc_transaction_all_amt_mean`, `mobile_eft_all_cnt_sum` | Tüm geçmiş üzerinden aggregation |
| **Trend (Son 3 Ay)** | `cc_transaction_all_amt_sum_last3m` | Yakın dönem davranışı |
| **Oran** | `cc_amt_ratio_last3m_vs_total` | Son 3 ay / toplam — en güçlü sinyal |
| **Demografik** | `tenure`, `age`, `work_type` | Müşteri profili |
| **Gruplandırılmış** | `sector_group`, `province_group` | Yüksek kardinalite → Top 10 + Diğer |

---

## 📈 Yarışma Metriği

Organizatörlerin resmi skoru üç bileşenden oluşmaktadır:
```
Skor = 0.40 × (Gini / Baseline_Gini)
     + 0.30 × (Recall@10% / Baseline_Recall@10%)
     + 0.30 × (Lift@10% / Baseline_Lift@10%)
```

> Baseline skoru aşmak (skor > 1.0) için Gini, üst %10'daki doğru yakalanma oranı ve lift birlikte optimize edildi.

---

## ⚙️ Model

| Parametre | Değer |
|---|---|
| Algoritma | XGBoost (`binary:logistic`) |
| Dengesizlik | `scale_pos_weight` = Non-Churn / Churn |
| Optimizasyon | Optuna (50 trial, resmi metrik üzerinde) |
| Eval metrik | AUC + Early Stopping |

---

## ⚠️ Limitasyonlar & Öğrenilen Dersler

İlk datathon deneyimi olması nedeniyle geriye dönük fark edilen bazı iyileştirme alanları:

- **Cross-validation eksikliği** — Tek bir 80/20 split kullanıldı; `StratifiedKFold` CV daha kararlı ve güvenilir sonuç üretirdi
- **Final model early stopping stratejisi** — Nihai modelde `eval_set` olarak test seti kullanıldı; ideal yaklaşım ayrı bir validation seti ile yapılmalıydı
- **Markdown dokümantasyonu** — Bulgular ve kararlar yalnızca kod yorumlarında kaldı; bir sonraki projede her adım ayrı markdown hücreleriyle belgelenecek

> Bu limitasyonları açıkça belgelemek, bir sonraki projede daha sağlam bir başlangıç yapabilmek için önemli. Hatalardan öğrenmek de sürecin bir parçası.

---

## 🚀 Kullanılan Teknolojiler

![Python](https://img.shields.io/badge/Python-3.10-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7-orange)
![Optuna](https://img.shields.io/badge/Optuna-3.0-purple)
![Pandas](https://img.shields.io/badge/Pandas-2.0-lightblue)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-1.3-red)

---

## 📁 Proje Yapısı
```
churn-analysis-inghubs/
│
├── churn-analysis-inghubs.ipynb   # Ana notebook
└── README.md
```
