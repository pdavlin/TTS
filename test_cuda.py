import torch
x = torch.rand(5, 3)
print(x)
print(torch.cuda.is_available())
torch.tensor(1).cuda()
print(torch.version.cuda)