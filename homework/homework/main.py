# pip install datasets

from datasets import load_dataset
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
import numpy as np
from rouge_score import rouge_scorer
import re
import warnings
import gc
import os
warnings.filterwarnings('ignore')

# GPU kullanımı için ayarlar
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # İlk GPU'yu kullan
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def clear_gpu_memory():
    """GPU belleğini temizle"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def check_gpu():
    """GPU durumunu kontrol et"""
    if not torch.cuda.is_available():
        print("❌ CUDA destekli GPU bulunamadı!")
        print("🔧 Sistem bilgileri:")
        print(f"   - PyTorch sürümü: {torch.__version__}")
        print(f"   - CUDA sürümü: {torch.version.cuda if torch.version.cuda else 'Yok'}")
        print(f"   - GPU sayısı: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
        raise RuntimeError("CUDA destekli GPU gerekli!")
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    print(f"✅ GPU tespit edildi: {gpu_name}")
    print(f"   - GPU Belleği: {gpu_memory:.1f} GB")
    print(f"   - CUDA Sürümü: {torch.version.cuda}")
    print(f"   - PyTorch Sürümü: {torch.__version__}")
    
    return gpu_memory

def get_optimal_batch_size(gpu_memory):
    """GPU belleğine göre optimal batch size belirle"""
    if gpu_memory < 4:
        return 1, 1, True  # train_batch, eval_batch, fp16
    elif gpu_memory < 6:
        return 2, 2, True
    elif gpu_memory < 8:
        return 4, 4, True
    else:
        return 8, 8, True

class NewsSummarizer:
    def __init__(self, model_name="t5-small", max_input_length=512, max_target_length=128):
        self.model_name = model_name
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        
        # GPU kontrolü
        self.gpu_memory = check_gpu()
        self.device = torch.device("cuda:0")
        torch.cuda.set_device(0)
        
        clear_gpu_memory()
        print("📥 Model GPU'ya yükleniyor...")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        
        # Modeli GPU'ya yükle
        self.model = self.model.to(self.device)
        
        print("✅ Model GPU'ya yüklendi")
        print(f"🎯 Model cihazı: {next(self.model.parameters()).device}")
        
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def preprocess_text(self, text):
        if not text or not isinstance(text, str):
            return ""
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
        text = text.lower().strip()
        if len(text.split()) < 10:
            return ""
        return text

    def tokenize_function(self, examples):
        articles = [self.preprocess_text(article) for article in examples['article']]
        highlights = [self.preprocess_text(highlight) for highlight in examples['highlights']]
        valid_indices = [i for i, (art, high) in enumerate(zip(articles, highlights)) 
                        if art and high and len(art.split()) > 10 and len(high.split()) > 3]
        if not valid_indices:
            return {
                "input_ids": [[0] * self.max_input_length],
                "attention_mask": [[1] * self.max_input_length],
                "labels": [[0] * self.max_target_length]
            }
        articles = [articles[i] for i in valid_indices]
        highlights = [highlights[i] for i in valid_indices]
        model_inputs = self.tokenizer(
            articles,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                highlights,
                max_length=self.max_target_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def prepare_data(self, train_dataset, validation_dataset, test_dataset):
        print("📊 Veri tokenization başlıyor...")
        # GPU belleğine göre batch size ayarla
        batch_size = 8 if self.gpu_memory >= 4 else 4
        train_dataset = train_dataset.map(self.tokenize_function, batched=True, batch_size=batch_size, remove_columns=train_dataset.column_names)
        validation_dataset = validation_dataset.map(self.tokenize_function, batched=True, batch_size=batch_size, remove_columns=validation_dataset.column_names)
        test_dataset = test_dataset.map(self.tokenize_function, batched=True, batch_size=batch_size, remove_columns=test_dataset.column_names)
        print("✅ Veri tokenization tamamlandı")
        return train_dataset, validation_dataset, test_dataset

    def train_model(self, train_dataset, validation_dataset, epochs=3):
        # GPU belleğine göre optimal parametreleri belirle
        train_batch, eval_batch, fp16 = get_optimal_batch_size(self.gpu_memory)
        
        print(f"🎯 GPU Eğitim parametreleri:")
        print(f"   - Train batch size: {train_batch}")
        print(f"   - Eval batch size: {eval_batch}")
        print(f"   - FP16: {fp16}")
        print(f"   - Epochs: {epochs}")
        
        training_args = Seq2SeqTrainingArguments(
            output_dir="./news_summarizer_model",
            eval_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=500,
            learning_rate=3e-5,
            per_device_train_batch_size=train_batch,
            per_device_eval_batch_size=eval_batch,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=epochs,
            predict_with_generate=True,
            fp16=fp16,
            report_to=None,
            logging_steps=100,
            warmup_steps=200,
            gradient_accumulation_steps=4,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            dataloader_num_workers=0,
            optim="adamw_torch",
        )
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            tokenizer=self.tokenizer,
        )
        trainer.train()
        clear_gpu_memory()
        trainer.save_model()
        self.tokenizer.save_pretrained("./news_summarizer_model")
        return trainer

    def generate_summary(self, text):
        # Modelin GPU'da olduğunu kontrol et
        if next(self.model.parameters()).device.type != 'cuda':
            print("⚠️  Model GPU'da değil! GPU'ya taşınıyor...")
            self.model = self.model.cuda()
        
        text = self.preprocess_text(text)
        if not text:
            return "Metin çok kısa veya geçersiz."
        
        inputs = self.tokenizer(
            text,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                summary_ids = self.model.generate(
                    inputs["input_ids"],
                    max_length=self.max_target_length,
                    min_length=10,
                    num_beams=2,
                    length_penalty=0.8,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                    do_sample=True,
                    top_k=50,
                    top_p=0.9,
                    temperature=0.7,
                )
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        if not summary or summary.strip() == "":
            return "Özet üretilemedi."
        return summary

    def evaluate_rouge(self, test_dataset, num_samples=5):
        print(f"🔍 GPU ile ROUGE değerlendirmesi başlıyor... ({num_samples} örnek)")
        rouge_scores = []
        for i in range(min(num_samples, len(test_dataset))):
            original_text = test_dataset[i]['article']
            target_summary = test_dataset[i]['highlights']
            predicted_summary = self.generate_summary(original_text)
            scores = self.scorer.score(target_summary, predicted_summary)
            rouge_scores.append(scores)
            print(f"📊 Örnek {i+1}:")
            print(f"Orijinal Metin: {original_text[:200]}...")
            print(f"Hedef Özet: {target_summary}")
            print(f"Üretilen Özet: {predicted_summary}")
            print(f"ROUGE-1: {scores['rouge1'].fmeasure:.4f}")
            print(f"ROUGE-2: {scores['rouge2'].fmeasure:.4f}")
            print(f"ROUGE-L: {scores['rougeL'].fmeasure:.4f}")
            print("-" * 50)
        avg_rouge1 = np.mean([score['rouge1'].fmeasure for score in rouge_scores])
        avg_rouge2 = np.mean([score['rouge2'].fmeasure for score in rouge_scores])
        avg_rougeL = np.mean([score['rougeL'].fmeasure for score in rouge_scores])
        print(f"\nOrtalama ROUGE Skorları:")
        print(f"ROUGE-1: {avg_rouge1:.4f}")
        print(f"ROUGE-2: {avg_rouge2:.4f}")
        print(f"ROUGE-L: {avg_rougeL:.4f}")
        return {
            'rouge1': avg_rouge1,
            'rouge2': avg_rouge2,
            'rougeL': avg_rougeL
        }

def create_sample_dataset():
    """Örnek veri seti oluştur"""
    sample_data = {
        'article': [
            "The United States has announced new economic sanctions against several countries. The sanctions target specific industries and individuals believed to be involved in illegal activities. Officials say the measures are necessary to protect national security interests. The move comes after months of diplomatic negotiations failed to achieve the desired results.",
            "Scientists have discovered a new species of marine life in the Pacific Ocean. The creature, which resembles a jellyfish, was found at depths of over 3,000 meters. Researchers believe this discovery could lead to breakthroughs in marine biology and our understanding of deep-sea ecosystems. The finding was made during a research expedition funded by international organizations.",
            "A major technology company has released its latest smartphone model. The new device features advanced camera technology and improved battery life. The company expects strong sales in the coming quarter. The phone includes several innovative features that set it apart from competitors in the market.",
            "Climate change researchers have published a new report showing alarming trends in global temperature increases. The study, conducted over five years, indicates that current efforts to reduce carbon emissions are insufficient. Scientists warn that immediate action is needed to prevent catastrophic environmental consequences.",
            "A breakthrough in renewable energy technology has been announced by researchers at a leading university. The new solar panel design achieves unprecedented efficiency levels while reducing manufacturing costs. This development could accelerate the transition to clean energy sources worldwide."
        ],
        'highlights': [
            "US announces new economic sanctions targeting specific industries and individuals for national security.",
            "New marine species discovered in Pacific Ocean at 3,000 meter depths, could advance marine biology research.",
            "Technology company releases new smartphone with advanced camera and battery technology, expects strong sales.",
            "Climate change report shows insufficient carbon reduction efforts, immediate action needed to prevent catastrophe.",
            "Breakthrough solar panel technology achieves high efficiency with lower costs, could accelerate clean energy transition."
        ]
    }
    
    from datasets import Dataset
    return Dataset.from_dict(sample_data)

def main():
    print("🚀 Haber Başlıklarından Otomatik Özetleme Sistemi (GPU Versiyonu)")
    print("=" * 60)
    
    try:
        # Örnek veri seti oluştur
        print("📊 Örnek veri seti oluşturuluyor...")
        sample_dataset = create_sample_dataset()
        
        # Veri setini böl
        train_size = int(0.7 * len(sample_dataset))
        val_size = int(0.15 * len(sample_dataset))
        test_size = len(sample_dataset) - train_size - val_size
        
        train_dataset = sample_dataset.select(range(train_size))
        validation_dataset = sample_dataset.select(range(train_size, train_size + val_size))
        test_dataset = sample_dataset.select(range(train_size + val_size, len(sample_dataset)))
        
        print(f"Train örnek sayısı: {len(train_dataset)}")
        print(f"Validation örnek sayısı: {len(validation_dataset)}")
        print(f"Test örnek sayısı: {len(test_dataset)}")
        
        summarizer = NewsSummarizer()
        print("✅ Model başarıyla yüklendi")
        
        print("📊 Veri hazırlanıyor...")
        train_data, val_data, test_data = summarizer.prepare_data(train_dataset, validation_dataset, test_dataset)
        print("✅ Veri hazırlandı")
        
        print("🎯 Model eğitimi başlıyor...")
        trainer = summarizer.train_model(train_data, val_data, epochs=3)
        print("✅ Model eğitimi tamamlandı")
        
        print("📈 Model değerlendirmesi başlıyor...")
        rouge_scores = summarizer.evaluate_rouge(test_dataset, num_samples=len(test_dataset))
        print("✅ Model değerlendirmesi tamamlandı")
        
    except Exception as e:
        print(f"❌ Hata: {e}")
        print("🔧 Çözüm önerileri:")
        print("   1. NVIDIA GPU sürücülerini güncelleyin")
        print("   2. CUDA Toolkit'i yükleyin")
        print("   3. PyTorch CUDA sürümünü yükleyin")
        print("   4. GPU'nuzun CUDA desteklediğinden emin olun")
        raise
    
    print("\n🎯 EK ÖRNEKLER:")
    sample_texts = [
        "The United States has announced new economic sanctions against several countries. The sanctions target specific industries and individuals believed to be involved in illegal activities. Officials say the measures are necessary to protect national security interests.",
        "Scientists have discovered a new species of marine life in the Pacific Ocean. The creature, which resembles a jellyfish, was found at depths of over 3,000 meters. Researchers believe this discovery could lead to breakthroughs in marine biology.",
        "A major technology company has released its latest smartphone model. The new device features advanced camera technology and improved battery life. The company expects strong sales in the coming quarter."
    ]
    for i, text in enumerate(sample_texts, 1):
        try:
            summary = summarizer.generate_summary(text)
            print(f"📝 Örnek {i}:")
            print(f"Giriş Metni: {text}")
            print(f"Üretilen Özet: {summary}")
            print("-" * 50)
        except Exception as e:
            print(f"❌ Örnek {i} işlenirken hata: {e}")
            break

if __name__ == "__main__":
    main()
