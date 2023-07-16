# PyTorch模型开发与部署基础平台介绍:

## 模型开发与部署:
### AI 硬件平台:
- Nvidia GPU
- Apple NPU
- Google TPU
- Intel VPU

### 服务端AI训练硬件:
- Nvidia 的 cuda
- google cloud 上的 TPU
- 华为 2018 推出了 Ascend910

### 终端AI推理芯片:
- 类电脑的设备、手机、汽车、摄像头、IoT、众多嵌入式设备
  - Nvidia：CUDA GPU，面相嵌入式的 JETSON
  - Intel：Movidius VPU（NCS2）
  - Apple：A12处理器（及之后）上的NPU
  - 高通：骁龙处理器
  - 华为：麒麟处理器（达芬奇架构）
  - 在桌面级设备：NVIDIA 的 tesla 系列和 GeForce 系列显卡
  - iOS & Android：高通骁龙、华为海思麒麟、联发科天玑等
  - 嵌入式：Nvidia、Intel
  - 此外，Google 的 Edge TPU、国产的寒武纪、百度昆仑、阿里含光800等

### 终端AI前向软件框架
- 桌面级（被NVIDIA CUDA设备垄断）上使用的是PyTorch、Tensorflow
- iOS上使用的是Apple的CoreML、PyTorch库等
- Android上使用的是TFlite框架，PyTorch库、NCNN库等
- Intel NCS上使用的是Intel的NCSDK
- Nvidia嵌入式设备上使用的是TensorRT

### 如何在终端部署 PyTorch 模型
- PyTorch的C++接口官方包名Lib Torch
- IOS：PyTorch -> ONNX -> CoreML -> iOS
- Android
  - PyTorch -> onnx -> ncnn -> android
  - PyTorch -> onnx -> tensorflow -> android

### 如何在服务器部署PyTorch模型：
Flask & Django -> NVIDIA Triton -> TorchServe

## TorchScript

### 介绍：
- TorchScript 是一种从 PyTorch 代码创建可序列化和可优化模型的方法
- 任何 TorchScript 程序都可以从 Python 进程中保存，并加载到没有 Python 依赖的进程中
- Torch Script 中的核心数据结构是 ScriptModule，它是 Torch 的 nn.Module 的类似物，代表整个模型作为子模块树
- 两种方式实现：tracing 和 Scripting

### Tracing
- Tracing仅仅正确地记录那些不是数据依赖的函数和 nn.Module （例如没有对数据的条件判断）并且它们也没有任何未跟踪的外部依赖（例如执行输入输出或访问全局变量）
- 由于跟踪仅记录张量上的操作，因此它不会记录任何控制流操作，如 if 语句或循环
- traced_foo = torch.jit.trace(foo, (torch.rand(3), torch.rand(3)))

### Scripting
- 可以使用 Python 语法直接编写 Torch Script 代码
- 可以在 ScriptModule 的子类上使用 torch.jit.scipt 批注（对于函数）或 torch.jit.script_method 批注（对于方法）来执行此操作

## TorchServer

### 介绍：
- 简化模型部署过程的一-种方法是使用模型服务器，即专门]设计用于在生产中提供机器学习预测的现成的Web应用程序。
- 模型服务器可轻松加载一个或多 个模型，并自动创建由可扩展Web服. 务器提供支持的预测API。
- 模型服务器还可以根据预测请求运行代码预处理和后处理。
- 模型服务器还提供对生产至关重要的功能,例如日志记录、监控和安全性等。
- 广为使用的模型服务器有TensorFlow Serving和Triton等

### TorchServe
#### 介绍：
- TorchServe由AWS和Facebook合作推出，并作为PyTorch开源项目的一部分提供。
- 通过TorchServe, PyTorch 现在可以更快地将其模型投入生产，而无需编写自定义代码
- TorchServe支持任何机器学习环境，包括Amazon SageMaker.容器服务和Amazon Elastic Compute Cloud (EC2)。
- TorchServe由Java实现，因此需要最新版本的OpenJDK来运行。
#### TorchServe 特性
- 提供低延迟预测API
- 为诸如对象检测和文本分类等常用应用程序嵌入了默认处理程序
- 多模型服务
- A/B测试模型版本控制
- 监控指标
- 用于集成应用程序的 RESTful 终端节点


## ONNX
- Open Neural Network Exchange (开放神经网络交换)格式，是一一个用于表示深度学习模型的标准r可使模型在不同框架之间进行转移
- 支持加载ONNX模型并进行推理的深度学习框架有: Caffe2, PyTorch,MXNet，ML.NET, TensorRT 和Microsoft CNTK，TensorFlow 也非官方的支持ONNX
- https://github.com/onnx/onnx
- https://github.com/onnx/models
- https://github.com/onnx/onnx-tensorflow
- 可视化工具: netron


