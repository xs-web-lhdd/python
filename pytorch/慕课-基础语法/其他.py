"""
torch.nn 库：
1、torch.nn 专门为神经网络设计的模块化接口
2、nn构建于 autograd 之上（自动实现了前向运算和反向传播），可以用来定义和运行神经网络
    nn.Parameter
    nn.Linear & nn.conv2d
    nn.functional
    nn.Module
    nn.Sequential

nn.Parameter：
    1、定义可训练参数
    2、self.my_param = nn.Parameter(torch.randn(1))   定义初始化可训练参数
    3、self.register_parameter                        注册可训练参数
    4、nn.ParameterList & nn.ParameterDict            定义字典或列表结构的多个可训练参数
       self.params = nn.ParameterList([nn.Parameter(torch.randn(10, 10)) for i in range(10)])
       self.params = nn.ParameterDict({
            'left': nn.Parameter(torch.randn(5,10)),
            'right': nn.Parameter(torch.randn(5,10))
       })

nn.Linear & nn.conv2d & nn.ReLU & nn.MaxPool2d(2) & nn.MSELoss 等等
1、各种神经网络层的定义，继承于 nn.Module 的子类
    self.conv1 = nn.Conv2d(1, 6, (5, 5))  是一个类
    调用时：self.conv1(x)
2、参数为 parameter 类型
    layer = nn.Linear(1, 1)
    layer.weight = nn.Parameter(torch.FloatTensor([[0]]))
    layer.bias = nn.Parameter(torch.FloatTensor([0]))

nn.functional：
1、包含 torch.nn 库中所有函数，包含大量 loss 和 activation function
    torch.nn.functional.conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)  是一个函数
2、nn.functional.xxx 是函数接口
3、nn.functional.xxx 无法与 nn.Sequential 结合使用
4、没有学习参数的等根据啊个人选择使用 nn.functional.xx 或 nn.Xxx
5、需要特别注意 dropout 层

nn 与 nn.functional 有什么区别？
1、nn.functional.xxx 是函数接口
2、nn.Xxx 是 nn.functional.xxx 的类封装，并且 nn.Xxx 都继承于一个共同祖先 nn.Module
3、nn.Xxx 除了具有 nn.functional.xxx 功能外，内部附带 nn.Module 相关的属性和方法，eg: train(), eval(), load_state_dict, state_dict
"""
