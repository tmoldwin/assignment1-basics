import torch
import einops
torch.device('cuda')
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version built with: {torch.version.cuda}")
try:
    assert torch.cuda.is_available()
    print("CUDA is available")
except AssertionError as e:
    print("CUDA is not available")
    print(e)

k = torch.tensor([[1,2,3],[4,5,6]])
print(k.device)
k = k.to(torch.device('cuda'))
print(k.device)
pty = torch.empty(3,3)
print(k)
print(pty)
y = k[0]
print(y)
print(k.stride)
k.view(1,6)
a = torch.tensor([[[1,2],[5,6]],[[7,8],[9,10]]])
b = torch.tensor([[7,8,9],[10,11,12]])
print(a.matmul(b))
c = einops.rearrange(a, 'hi mom you -> (you hi mom)')
print(a@b)
print(c)