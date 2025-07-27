# Haber Başlıklarından Otomatik Özetleme Sistemi (GPU Versiyonu)

## Proje Açıklaması

Bu proje, örnek veri seti kullanarak transformer tabanlı bir haber özetleme sistemi geliştirmeyi amaçlamaktadır. T5-small modeli kullanılarak article alanından summary alanını tahmin eden bir model eğitilmiştir. **Bu versiyon NVIDIA GPU ile çalışacak şekilde optimize edilmiştir.**

## Kullanılan Teknolojiler

- **Model**: T5-small (Text-to-Text Transfer Transformer)
- **Kütüphaneler**: Hugging Face Transformers, PyTorch, Datasets
- **Değerlendirme Metrikleri**: ROUGE-1, ROUGE-2, ROUGE-L
- **Veri Seti**: Örnek haber veri seti (5 örnek)
- **Donanım**: NVIDIA GPU optimizasyonu

## Sistem Gereksinimleri

### GPU Gereksinimleri
- **NVIDIA GPU** (CUDA destekli)
- **En az 4GB GPU belleği** (önerilen: 6GB+)
- **CUDA Toolkit 11.0+**
- **PyTorch CUDA sürümü**

### Desteklenen GPU'lar
- GeForce RTX/GTX serisi
- Quadro serisi
- Tesla serisi

## Özellikler

### 1. Veri Ön İşleme
- HTML etiketlerinin temizlenmesi
- Fazla boşlukların kaldırılması
- Küçük harfe çevirme
- Maksimum uzunluk sınırlamaları (Input: 512, Target: 128 token)
- Sequence truncation ve padding

### 2. Model Kurulumu
- T5-small modeli (düşük donanım gereksinimleri)
- Hugging Face Transformers kütüphanesi
- Seq2SeqTrainer ile eğitim
- GPU optimizasyonu

### 3. Eğitim Parametreleri (GPU Optimizasyonu)
- **Epoch sayısı**: 3 (hızlı prototip için)
- **Learning rate**: 3e-5
- **Batch size**: GPU belleğine göre otomatik ayarlanır
  - 4GB altı: 1
  - 4-6GB: 2
  - 6-8GB: 4
  - 8GB+: 8
- **Weight decay**: 0.01
- **FP16**: Açık (GPU için)
- **Mixed Precision**: Açık

### 4. Değerlendirme
- ROUGE skorları (ROUGE-1, ROUGE-2, ROUGE-L)
- İnsan gözüyle değerlendirme için örnek çıktılar
- Test seti üzerinde performans ölçümü

## Kurulum ve Çalıştırma

### Gereksinimler
```bash
pip install -r requirements.txt
```

### GPU Kontrolü
```bash
python -c "import torch; print(f'CUDA mevcut: {torch.cuda.is_available()}'); print(f'GPU sayısı: {torch.cuda.device_count()}'); print(f'GPU adı: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"Yok\"}')"
```

### Çalıştırma
```bash
python main.py
```

### Test Modu
```bash
python test_summarizer.py
```

## Proje Yapısı

```
homework/
├── main.py              # Ana uygulama dosyası (GPU versiyonu)
├── test_summarizer.py   # Test dosyası
├── requirements.txt     # Gerekli kütüphaneler
├── README.md           # Bu dosya
└── news_summarizer_model/  # Eğitilen model (otomatik oluşur)
```

## Geliştirme Süreci

### 1. Veri Hazırlama
- Örnek haber veri seti oluşturuldu (5 örnek)
- Veri seti train/validation/test olarak bölündü
- Metin temizleme ve tokenization işlemleri uygulandı

### 2. Model Seçimi
- T5-small modeli seçildi (düşük donanım gereksinimleri)
- Seq2Seq mimarisi kullanıldı
- Hugging Face Transformers kütüphanesi tercih edildi
- GPU optimizasyonu yapıldı

### 3. Eğitim Süreci
- 3 epoch ile hızlı prototip alındı
- GPU belleğine göre otomatik parametre ayarlama
- Validation seti üzerinde değerlendirme
- Model checkpoint'leri kaydedildi

### 4. Değerlendirme
- ROUGE metrikleri hesaplandı
- Örnek çıktılar incelendi
- Performans analizi yapıldı

## Sonuçlar

### ROUGE Skorları (Beklenen)
- ROUGE-1: ~0.35-0.45
- ROUGE-2: ~0.15-0.25
- ROUGE-L: ~0.25-0.35

### Örnek Çıktılar
Sistem, verilen haber metinlerinden anlamlı özetler üretebilmektedir. Örnekler main.py çalıştırıldığında görülebilir.

## GPU Optimizasyonu

### Otomatik Batch Size Ayarlama
| GPU Belleği | Batch Size | Açıklama |
|-------------|------------|----------|
| < 4GB | 1 | Küçük GPU'lar için |
| 4-6GB | 2 | Orta boy GPU'lar için |
| 6-8GB | 4 | Büyük GPU'lar için |
| 8GB+ | 8 | Çok büyük GPU'lar için |

### Performans
- Eğitim süresi: ~2-5 dakika (GPU'ya bağlı)
- Bellek kullanımı: GPU belleğine göre değişir
- Disk kullanımı: ~500 MB

## Sorun Giderme

### GPU Bulunamadı Hatası
```bash
# NVIDIA sürücülerini kontrol et
nvidia-smi

# PyTorch CUDA sürümünü kontrol et
python -c "import torch; print(torch.version.cuda)"

# CUDA Toolkit'i yeniden yükle
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Bellek Yetersiz Hatası
- Batch size'ı manuel olarak azaltın
- Gradient accumulation artırın
- Model boyutunu küçültün (T5-tiny)

## Gelecek Geliştirmeler

1. **Veri Seti Genişletme**: Daha fazla örnek ile eğitim
2. **Model Büyütme**: T5-base kullanımı (daha fazla GPU belleği gerekir)
3. **Hiperparametre Optimizasyonu**: Grid search ile en iyi parametrelerin bulunması
4. **Türkçe Veri Seti**: Türkçe haber özetleme için uyarlama
5. **Web Arayüzü**: Kullanıcı dostu web uygulaması

## Teknik Detaylar

### Model Mimarisi
- **Encoder-Decoder**: T5 transformer mimarisi
- **Attention Mechanism**: Multi-head self-attention
- **Vocabulary Size**: ~32,000 token
- **Model Parameters**: ~60M (T5-small)

### Eğitim Detayları
- **Optimizer**: AdamW
- **Scheduler**: Linear warmup
- **Mixed Precision**: FP16 (GPU)
- **Gradient Clipping**: 1.0
- **Device**: CUDA GPU

## Katkıda Bulunanlar

Bu proje NLP dersi kapsamında geliştirilmiştir.

## Lisans

Bu proje eğitim amaçlı geliştirilmiştir. 