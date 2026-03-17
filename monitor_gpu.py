
import torch
import time

def monitor_gpu():
    """Мониторинг использования GPU памяти"""
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Allocated: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")
        print(f"Cached: {torch.cuda.memory_reserved(0)/1024**3:.2f} GB")
        print(f"Free: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0))/1024**3:.2f} GB")
    else:
        print("CUDA not available")

if __name__ == "__main__":
    monitor_gpu()