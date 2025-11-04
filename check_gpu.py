import torch

# Check PyTorch CUDA
print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"CUDA device name: {torch.cuda.get_device_name()}")

    # Check GPU memory using PyTorch
    print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
    print(f"GPU Memory Reserved: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")
    print(f"GPU Memory Total: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.1f} MB")

    # Check if we can allocate some GPU memory
    try:
        # Try to allocate 1GB of GPU memory
        test_tensor = torch.zeros((256, 1024, 1024), dtype=torch.float32, device="cuda")
        print("Successfully allocated ~1GB test tensor on GPU")
        print(f"GPU Memory Allocated after test: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        del test_tensor
        torch.cuda.empty_cache()
        print(f"GPU Memory Allocated after cleanup: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
    except Exception as e:
        print(f"Failed to allocate GPU memory: {e}")
else:
    print("CUDA not available")
