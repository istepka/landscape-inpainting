import torch

# Check if CUDA (GPU support) is available
if torch.cuda.is_available():
    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()
    
    print(f"PyTorch can see {num_gpus} GPU(s).")
    
    # Print information about each GPU
    for gpu_id in range(num_gpus):
        print(f"GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
else:
    print("PyTorch cannot find any GPUs. Make sure you have installed the appropriate GPU drivers.")
