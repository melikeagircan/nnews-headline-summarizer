#!/usr/bin/env python3
"""
CNN/DailyMail Veri Seti İndirme Scripti
Bu script CNN/DailyMail veri setini manuel olarak indirir.
"""

import os
import sys
from datasets import load_dataset, DatasetDict
import json

def download_cnn_dailymail():
    """CNN/DailyMail veri setini indir"""
    print("CNN/DailyMail veri seti indiriliyor...")
    
    try:
        # Veri setini indir
        dataset = load_dataset("cnn_dailymail", "3.0.0", cache_dir="./data_cache")
        print("✅ CNN/DailyMail 3.0.0 başarıyla indirildi!")
        
    except Exception as e:
        print(f"❌ CNN/DailyMail 3.0.0 indirilemedi: {e}")
        
        try:
            dataset = load_dataset("cnn_dailymail", "2.0.0", cache_dir="./data_cache")
            print("✅ CNN/DailyMail 2.0.0 başarıyla indirildi!")
            
        except Exception as e2:
            print(f"❌ CNN/DailyMail 2.0.0 indirilemedi: {e2}")
            
            try:
                dataset = load_dataset("cnn_dailymail", cache_dir="./data_cache")
                print("✅ CNN/DailyMail default başarıyla indirildi!")
                
            except Exception as e3:
                print(f"❌ CNN/DailyMail default indirilemedi: {e3}")
                print("\n🔧 Manuel indirme gerekli!")
                return None
    
    # Veri seti bilgilerini göster
    print(f"\n📊 Veri Seti Bilgileri:")
    print(f"  Train: {len(dataset['train'])} örnek")
    print(f"  Validation: {len(dataset['validation'])} örnek")
    print(f"  Test: {len(dataset['test'])} örnek")
    print(f"  Sütunlar: {dataset['train'].column_names}")
    
    # Örnek veri göster
    print(f"\n📝 Örnek Veri:")
    sample = dataset['train'][0]
    print(f"  Article: {sample['article'][:100]}...")
    print(f"  Highlights: {sample['highlights']}")
    
    return dataset

def save_sample_data(dataset, num_samples=100):
    """Örnek veriyi JSON dosyasına kaydet"""
    print(f"\n💾 {num_samples} örnek veri kaydediliyor...")
    
    sample_data = {
        'train': dataset['train'].select(range(min(num_samples, len(dataset['train'])))).to_dict(),
        'validation': dataset['validation'].select(range(min(num_samples//5, len(dataset['validation'])))).to_dict(),
        'test': dataset['test'].select(range(min(num_samples//5, len(dataset['test'])))).to_dict()
    }
    
    with open('./cnn_dailymail_sample.json', 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)
    
    print("✅ Örnek veri 'cnn_dailymail_sample.json' dosyasına kaydedildi!")

def main():
    print("=" * 60)
    print("CNN/DailyMail Veri Seti İndirme Aracı")
    print("=" * 60)
    
    # Veri setini indir
    dataset = download_cnn_dailymail()
    
    if dataset is not None:
        # Örnek veriyi kaydet
        save_sample_data(dataset, num_samples=100)
        
        print(f"\n🎉 İndirme tamamlandı!")
        print(f"📁 Cache klasörü: ./data_cache")
        print(f"📄 Örnek veri: ./cnn_dailymail_sample.json")
        
    else:
        print(f"\n❌ İndirme başarısız!")
        print(f"🔗 Manuel indirme linki: https://huggingface.co/datasets/cnn_dailymail")
        print(f"📧 Alternatif: Veri setini manuel olarak indirip projeye ekleyin.")

if __name__ == "__main__":
    main() 