import subprocess
import sys
import os

def install_cuda_pytorch():
    print("=== CUDA PYTORCH KURULUMU ===")
    print()
    
    # Mevcut PyTorch'u kaldır
    print("1. Mevcut PyTorch kaldırılıyor...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "torch", "torchvision", "torchaudio", "-y"], 
                      check=True, capture_output=True)
        print("✅ Mevcut PyTorch kaldırıldı")
    except:
        print("⚠️ PyTorch kaldırılamadı (zaten yüklü olmayabilir)")
    
    print()
    print("2. CUDA destekli PyTorch kuruluyor...")
    print("Bu işlem birkaç dakika sürebilir...")
    
    # CUDA 11.8 için PyTorch kur
    try:
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "torch", "torchvision", "torchaudio", 
            "--index-url", "https://download.pytorch.org/whl/cu118"
        ], check=True, capture_output=True, text=True)
        
        print("✅ CUDA destekli PyTorch başarıyla kuruldu!")
        print()
        print("3. Kurulum test ediliyor...")
        
        # Test et
        test_result = subprocess.run([
            sys.executable, "-c", 
            "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU Count:', torch.cuda.device_count())"
        ], check=True, capture_output=True, text=True)
        
        print(test_result.stdout)
        
        if "CUDA: True" in test_result.stdout:
            print("🎉 GPU kullanıma hazır!")
            print("main.py dosyasını çalıştırabilirsiniz.")
        else:
            print("❌ GPU hala kullanılamıyor.")
            print("NVIDIA sürücülerini kontrol edin.")
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Kurulum hatası: {e}")
        print("Manuel kurulum için:")
        print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

if __name__ == "__main__":
    install_cuda_pytorch() 