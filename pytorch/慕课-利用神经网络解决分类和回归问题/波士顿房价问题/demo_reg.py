import torch
import numpy as np
import re

# 1、加载数据
ff = open('housing.data').readlines()
data = []

for item in ff:
    # 通过正则，将多个空格合并为一个空格
    out = re.sub(r"\s{2,}", " ", item).strip()
    data.append(out.split(" "))
data = np.array(data).astype(np.float64)
# print(data.shape)

Y = data[:, -1]
X = data[:, 0: -1]

# print(Y.shape, X.shape)

# 划分训练集和测试集
X_train = X[0: 496, ...]
Y_train = Y[0: 496, ...]
X_test = X[496:, ...]
Y_test = Y[496:, ...]

# print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)


# 2、定义网络：
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_output):
        super(Net, self).__init__()
        # 线性，回归模型
        self.hidden = torch.nn.Linear(n_feature, 128)
        self.predict = torch.nn.Linear(128, n_output)

    def forward(self, x):
        out = self.hidden(x)
        out = torch.relu(out)
        out = self.predict(out)
        return out


# 传入输入特征数量 13 输出特征数量 1
net = Net(13, 1)

# 3、定义损失：
# 采用均方损失
loss_func = torch.nn.MSELoss()

# 4、优化器：
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

# 5、定义训练过程：
for i in range(1000):
    x_data = torch.tensor(X_train, dtype=torch.float32)
    y_data = torch.tensor(Y_train, dtype=torch.float32)
    # 根据 x_data 求出预测值：
    pred = net.forward(x_data)
    # 删除维度：
    pred = torch.squeeze(pred)
    loss = loss_func(pred, y_data) * 0.001

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("ite：{}, loss：{}".format(i, loss))
    print(pred[0: 10])
    print(y_data[0: 10])

    # 6、进行测试
    x_data = torch.tensor(X_test, dtype=torch.float32)
    y_data = torch.tensor(Y_test, dtype=torch.float32)
    pred = net.forward(x_data)
    pred = torch.squeeze(pred)
    loss_test = loss_func(pred, y_data) * 0.001
    print("ite：{}, loss_test：{}".format(i, loss_test))


torch.save(net, 'model/model.pkl')
# torch.save(net.state_dict(), 'model/params.pkl')
