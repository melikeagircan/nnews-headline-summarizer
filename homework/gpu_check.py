import torch

print("=== GPU KONTROL SİSTEMİ ===")
print()

# PyTorch versiyonu
print(f"PyTorch Versiyonu: {torch.__version__}")

# CUDA mevcut mu?
cuda_available = torch.cuda.is_available()
print(f"CUDA Mevcut: {cuda_available}")

if cuda_available:
    # GPU sayısı
    gpu_count = torch.cuda.device_count()
    print(f"GPU Sayısı: {gpu_count}")
    
    # Her GPU'nun bilgileri
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
        print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    
    # Aktif GPU
    current_device = torch.cuda.current_device()
    print(f"Aktif GPU: {current_device}")
    
    # GPU belleği
    allocated = torch.cuda.memory_allocated() / 1024**3
    cached = torch.cuda.memory_reserved() / 1024**3
    print(f"Kullanılan Bellek: {allocated:.2f} GB")
    print(f"Rezerve Bellek: {cached:.2f} GB")
    
    print("\n✅ GPU kullanıma hazır!")
else:
    print("❌ CUDA destekli GPU bulunamadı!")
    print("Olası nedenler:")
    print("1. NVIDIA GPU yok")
    print("2. CUDA sürücüleri yüklü değil")
    print("3. PyTorch CUDA versiyonu yüklü değil")
    print("\nKurulum için:")
    print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118") 