import torch.nn as nn
from torchvision import models

net_name = 'pre_resnet'


class resnet18(nn.Module):
    def __init__(self):
        super(resnet18, self).__init__()
        # 加载预训练模型：
        self.model = models.resnet18(pretrained=True)
        # 重新定义输出层，因为这里是 10 分类，所以是 10
        self.num_feature = self.model.fc.in_features
        self.model.fc = nn.Linear(self.num_feature, 10)

    def forward(self, x):
        out = self.model(x)

        return out


def pytorch_resnet18():
    return resnet18()
