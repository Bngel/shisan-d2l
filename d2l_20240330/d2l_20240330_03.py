import torch

x = torch.arange(4.0)

print(x)

x.requires_grad_(True)
print(x.grad)

# y = 2 * x * x
# y' = 4 * x
y = 2 * torch.dot(x, x)
print(y)

y.backward()
print(x.grad)
print(x.grad == 4 * x)

x.grad.zero_()
y = x.sum()
y.backward()
print(x.grad)

x.grad.zero_()
y = x * x
# 等价于 y.backward(torch.ones(len(x)))
y.sum().backward()
print(x.grad)

x.grad.zero_()
y = x * x
u = y.detach()
z = u * x

z.sum().backward()
print(x.grad)

# 每一次对变量的计算, torch就会在变量中存下计算图
