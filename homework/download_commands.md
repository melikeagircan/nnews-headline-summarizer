# CNN/DailyMail Veri Seti İndirme Rehberi

## 🚀 Hızlı İndirme Komutları

### 1. Python Script ile İndirme
```bash
cd homework
python download_cnn_dailymail.py
```

### 2. Doğrudan Python Komutu ile İndirme
```bash
python -c "from datasets import load_dataset; dataset = load_dataset('cnn_dailymail', '3.0.0', cache_dir='./data_cache'); print('İndirme tamamlandı!')"
```

### 3. Hugging Face CLI ile İndirme
```bash
pip install huggingface_hub
huggingface-cli download cnn_dailymail --local-dir ./cnn_dailymail_data
```

## 📁 Manuel İndirme Adımları

### Adım 1: Hugging Face'den İndir
1. https://huggingface.co/datasets/cnn_dailymail adresine git
2. "Files and versions" sekmesine tıkla
3. "Download" butonuna tıkla
4. ZIP dosyasını indir

### Adım 2: Dosyaları Çıkar
```bash
# Windows
tar -xf cnn_dailymail-3.0.0.zip

# Linux/Mac
unzip cnn_dailymail-3.0.0.zip
```

### Adım 3: Projeye Ekle
```bash
# İndirilen dosyaları homework klasörüne kopyala
cp -r cnn_dailymail-3.0.0/* homework/
```

## 🔧 Alternatif İndirme Yöntemleri

### Yöntem 1: Git ile İndirme
```bash
git clone https://huggingface.co/datasets/cnn_dailymail
cd cnn_dailymail
```

### Yöntem 2: wget ile İndirme
```bash
wget https://huggingface.co/datasets/cnn_dailymail/resolve/main/dataset_info.json
wget https://huggingface.co/datasets/cnn_dailymail/resolve/main/cnn_dailymail-train.arrow
wget https://huggingface.co/datasets/cnn_dailymail/resolve/main/cnn_dailymail-validation.arrow
wget https://huggingface.co/datasets/cnn_dailymail/resolve/main/cnn_dailymail-test.arrow
```

### Yöntem 3: curl ile İndirme
```bash
curl -L -o cnn_dailymail.zip https://huggingface.co/datasets/cnn_dailymail/resolve/main/cnn_dailymail-3.0.0.zip
```

## 📊 Veri Seti Bilgileri

- **Boyut:** ~1.5 GB
- **Train:** ~287,113 örnek
- **Validation:** ~13,368 örnek  
- **Test:** ~11,490 örnek
- **Format:** Arrow (.arrow) dosyaları
- **Sütunlar:** article, highlights, id

## ⚠️ Sorun Giderme

### Hata: "Dataset not found"
```bash
# Datasets kütüphanesini güncelle
pip install --upgrade datasets

# Cache'i temizle
rm -rf ~/.cache/huggingface/
```

### Hata: "Permission denied"
```bash
# Windows için
python download_cnn_dailymail.py --user

# Linux/Mac için
sudo python download_cnn_dailymail.py
```

### Hata: "Out of memory"
```bash
# Daha küçük batch ile indir
python -c "from datasets import load_dataset; dataset = load_dataset('cnn_dailymail', '3.0.0', streaming=True)"
```

## 🎯 Başarılı İndirme Kontrolü

```python
from datasets import load_dataset

# Veri setini yükle
dataset = load_dataset("cnn_dailymail", "3.0.0")

# Kontrol et
print(f"Train: {len(dataset['train'])} örnek")
print(f"Validation: {len(dataset['validation'])} örnek")
print(f"Test: {len(dataset['test'])} örnek")

# Örnek göster
print(dataset['train'][0])
```

## 📞 Yardım

Eğer indirme sorunları yaşıyorsanız:
1. İnternet bağlantınızı kontrol edin
2. Disk alanınızı kontrol edin (en az 2GB boş alan)
3. Firewall ayarlarınızı kontrol edin
4. VPN kullanıyorsanız kapatmayı deneyin 