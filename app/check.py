import torch

print(torch.__version__)
print(torch.version.hip)

print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))

x = torch.rand(1000, 1000, device="cuda")
print(x.mean())

a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
b = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)

c = torch.matmul(a, b)

print(c[0][0])