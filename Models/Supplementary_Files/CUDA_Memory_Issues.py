import torch

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(torch.cuda.memory_summary(device=device, abbreviated=False))

# print(torch.cuda.memory_stats())


# import gc
# del variables
# gc.collect()


# torch.cuda.empty_cache()




# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print('Torch Version : ', torch.__version__)
# if device == torch.device('cuda'): print('CUDA Version  : ', torch.version.cuda)
# print('There are %d GPU(s) available.' % torch.cuda.device_count())
# print('We will use the GPU:', torch.cuda.get_device_name(0))




# nvidia-smi
# nvidia-smi | grep 'python' | awk '{ print $5 }' | xargs -n1 kill -9


nvidia-smi | grep 'Default' | awk '{ print $9 " out of "  $11 " - GPU Util: " $13}'