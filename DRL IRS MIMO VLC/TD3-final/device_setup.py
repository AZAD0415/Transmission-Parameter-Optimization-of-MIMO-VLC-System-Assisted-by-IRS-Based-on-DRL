import torch

def setup_device():
    """设置训练设备"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
        print_gpu_memory_status()
    else:
        device = torch.device("cpu")
        print("使用CPU")
    return device

def print_gpu_memory_status():
    """打印GPU内存状态"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
            allocated_memory = torch.cuda.memory_allocated(i) / 1024**3  # GB
            cached_memory = torch.cuda.memory_reserved(i) / 1024**3  # GB
            free_memory = total_memory - allocated_memory - cached_memory
            
            print(f"GPU {i} ({torch.cuda.get_device_name(i)}):")
            print(f"  总内存: {total_memory:.2f} GB")
            print(f"  已分配内存: {allocated_memory:.2f} GB")
            print(f"  缓存内存: {cached_memory:.2f} GB")
            print(f"  可用内存: {free_memory:.2f} GB") 