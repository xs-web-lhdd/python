import torch

"""
API：
torch.mean()    返回平均值
torch.sum()     返回总和
torch.prod()    计算所有元素的积
torch.max()     返回最大值
torch.min()     返回最小值
torch.argmax()  返回最大值排序的索引值
torch.argmin()  返回最小值排序的索引值
torch.std()     返回标准差
torch.var()     返回方差
torch.median()  返回中间值
torch.mode()    返回众数
torch.histc()   计算 input 的直方图
torch.bincount()返回每个值的频数
"""

# 代码演示：
a = torch.rand(2, 2)

print(a)
print(torch.mean(a, dim=0))
print(torch.sum(a, dim=0))
print(torch.prod(a, dim=0))
print(torch.max(a, dim=0))

print(torch.std(a))
print(torch.var(a))
print(torch.median(a))
print(torch.mode(a))


print('============= 频数 ==============')

b = torch.randint(0, 10, [10])
print(b)
print(torch.bincount(b))

# torch.manual_seed(seed)  用来保证随机抽样的结果一致

torch.manual_seed(1)
mean = torch.rand(1, 2)
std = torch.rand(1, 2)

print(torch.normal(mean, std))
