import torch

print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cudnn.version())
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_properties(0))

#Check if models are on CUDA
# next(self.model.parameters()).is_cuda
# next(self.model.parameters()).device