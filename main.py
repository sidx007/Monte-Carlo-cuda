import torch

if not torch.cuda.is_available():
    print("No GPU detected.")
else:
    gpu_count = torch.cuda.device_count()
    print(f"Number of GPUs: {gpu_count}\n")

    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Total Memory: {props.total_memory / (1024**3):.2f} GB")
        print(f"  Multiprocessors: {props.multi_processor_count}")
        print(f"  CUDA Cores: {props.multi_processor_count * 64}")  # approx
        print(f"  Compute Capability: {props.major}.{props.minor}")
        print("-" * 40)