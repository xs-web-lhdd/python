"""
Tensor 中的范数运算

范数：在泛函数分析中，它定义在赋范线性空间中，并满足一定的条件，即：
        1、非负性  2、齐次性  3、三角不等式
    常被用来度量某个向量空间（或矩阵）中的每个向量的长度或大小

0范数 / 1范数/ 2范数/ p范数 / 核范数
> torch.dist(input, other, p = 2) 计算 p 范数
> torch.norm() 计算 2 范数

0范数： 当前向量中非 0 元素的个数和
1范数： 向量元素中绝对值的和
2范数： 元素的平方和再进行开平方
p范数： 元素绝对值的 p 次方求和，然后再进行 1/p 次幂（开 p 次方）的结果
"""

import torch

a = torch.rand(1, 1)
b = torch.rand(1, 1)


print(a, b)

print(torch.dist(a, b, p=1))
print(torch.dist(a, b, p=2))
print(torch.dist(a, b, p=3))
