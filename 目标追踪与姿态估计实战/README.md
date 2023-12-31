# 目标追踪与姿态估计

## OpenPose
讲的比较好的知乎文章：https://zhuanlan.zhihu.com/p/360541947

### 什么是姿态估计？
- 得到人体各个关键点位置
- 将它们按顺序进行拼接
- 这其中难点是什么？ 1、遮挡现象  2、关键点怎么匹配（人多时左右肩膀点怎么匹配）
- OpenPose如何做的呢？

### 应用领域：
- 各种特效肯定离不开关键点信息（健身APP，检测姿态估计，判断姿势是否正确）
- 行为特征分析，需各位置信息（在矿井中检测头部的位置与安全帽的位置是否重叠，重叠了代表带安全帽了，没重叠代表没有带安全帽）
- 异常检测，突变现象（游泳馆泳池进行用户姿态分析，判断是否溺水，减少安全事故）
- 各类配准分析任务都需要它作为基础


### Top-down 方法两步走：
1、检测得到所有人的框
2、对每一个框进行姿态估计输出结果

#### Top-down 方法问题：
- 姿态估计做成啥样主要由于人体检测所决定，能检测到效果估计也没问题
- 但是如果俩人出现重叠，只能检测到一个人，那肯定会丢失一个目标
- 计算效率有点低，如果一张图像中存在很多人，那姿态估计得相当慢了（Top-down 只能做离线的，却做不了实时的）
- 能不能设计一种方法不依赖于人体框而是直接进行预测呢？

#### 一个恐怖的例子：
如何得到姿态估计结果呢？分几步走？
- 1、首先得到所有关键点的位置
- 2、图中有很多个人，我们需要把属于同一个人的拼接到一起

### bottom-up 挑战任务（首先得到所有关键点再拼接）
- 现在只检测了所有关键点位置，你能把他们拼起来嘛？

- 如果得到关键点位置：通过热度图（高斯）得到每一个关键点的预测结果

### 匹配方法：
- 如果同时考虑多种匹配，那太难了
- 咱们固定好就是二分图，这样可以直接套匈牙利算法


### Convolutional Pose Machines
- 为 OpenPose 后面的工作奠定了基础，也可以当作基础框架
- 通过多个 stage 来不断优化关键点位置（stage1 预测完全错误，2 和 3 在纠正）
- stage 越多相当于层数越深，模型感受野越大，姿态估计需要更大的感受野
- 每个 stage 都加损失函数，也就是中间过程才得做的好才行


### OpenPose
- 两个网络结构分别搞定： 1、关键点预测；2、姿势的“亲和力”向量
- 序列的作用： 多个 stage，相当于纠正的过程，不断调整预测结果
- 关键点预测：一口气预测所有关键点的特征图
- 整体框架：两个分支都要经过多个阶段，注意每个阶段后要把特征拼接一起


## Deep Sort 算法
知乎文章：https://zhuanlan.zhihu.com/p/133678626
源码：https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch

### 卡尔曼滤波：不管公式怎么推导的，科学家推导的
- 结合已知信息估计最优位置
- 本质是优化估计算法
- 例如估计人在下一帧的位置
- 阿波罗登月就用到了

- 任何状态都会受外部环境的影响（可以理解为噪音点），通常呈正态分布

- 本质上就是基于估计值和观测值进行综合（如下一帧预测值和下一帧检测值），对估计值和预测值分别设置权重值然后进行综合

- 其实就是两个分布（估计值、观测值）的乘积，得到更准确的估计（像是取交集）

- 两大核心模块：
  - 预测阶段：预测状态估计值及其协方差
    - 在单状态中，协方差矩阵就是其方差
    - 它就是预测状态中的不确定性的度量（噪声导致）
  - 更新阶段：后验估计 = 先验估计 + 卡尔曼增益 * （观测值 与 先验估计的差异）。在这里 先验估计是预测阶段的值，卡尔曼增益是权重值
    - 第二步要基于预测值更新参数
    - 例如追踪每一帧的状态肯定要变的
    - 预测完需要根据观测值来修正
    - 修正后的状态值去估计下一帧

- 卡尔曼增益：
  - 目的是让最优估计值的方差更小
  - 相当于一个权重项，该怎么利用估计与观测
  - 它就决定了卡尔曼滤波的核心作用


- 当观测没有噪音时（可以理解为观测值没有任何误差，就是实际值），那么最优估计等于观测值


- 追踪问题考虑的状态：
  - 均值（Mean）：8 维向量表示 x = [cx, cy, r, h, vx, vy, vr, vh]
  - 中心坐标（cx, cy），宽高比 r, 高 h，以及各自的速度变化值组成
  - 协方差矩阵：表示目标位置信息的不确定性，由 8 * 8 的矩阵表示


- 追踪过程也分为两个阶段： 每一个 track 都要预测下一时刻的状态，并基于检测到的结果来修正（匀速，线性，咱们追踪通常都是一帧一帧处理的，因为卡尔曼公式是基于线性做的）

### 匈牙利算法（匹配）：

#### 目的：
- 完成匹配的同时最小化代价矩阵
- 当时桢检测到的目标该匹配到前面哪一个 track 呢？
- 感觉有时候并不是最优分配，而是尽可能多的分配
- 目标追踪任务，detr 目标检测任务中都用到了该方法

#### 匈牙利算法举例：算法详情（了解即可，不需要背，不同情况用的算法也不一样，这里只是举例）：
- 如果代价矩阵的某一行或某一列同时加上或减去某个数，则这个新的代价矩阵的最优分配仍然是原代价矩阵的最优分配
- 1、对于矩阵的每一行，减去其中最小的元素
- 2、对于矩阵的每一列，减去其中最小的元素
- 3、用最少的水平线或垂直线覆盖矩阵中所有的 0
- 4、如果线的数量 = N，则找到了最优分配，算法结束，否则进入 5 
- 5、找到没有被任何线覆盖的最小元素，每个没被线覆盖的行减去这个元素，每个被线覆盖的列加上这个元素，返回步骤 3

#### 代码调用 匈牙利算法：只需要准备好代价矩阵传递给 API 即可
- 通过 sklearn 中 linear_assignment() API进行调用
- 通过 scipy 中 linear_sum_assignment() API进行调用

#### 构建代价矩阵：
- 运动信息匹配（卡尔曼估计）
- 外观匹配（ReID）
- IOU 匹配（BBOX）

##### ReID 特征：行人重识别
- 追踪人所以用到了 ReID（行人重识别训练好的模型），如果追踪其他目标需要自己训练
- 很简单的一个网络结构（不能太复杂，网络结构太复杂就可能做不到实时了），对输入的 bbox 进行特征提取，返回 128 维特征

- 根据当前检测的所有 bbox 与当前所有 track ，先得到所有 ReID 特征
- 当前每一个 track（跟踪器） 均存了一个特征序列（就是每一次匹配都会保留一份特征）
- 例如一个 track 有 5 份 128 维向量，选其与每个 bbox 余弦距离最小的作为输入
- track 保存的特征数量是有上限的，默认参数是 100 个



### sort 算法：
1、卡尔曼预测与更新
2、匈牙利匹配返回结果

将预测后的 tracks 和 当前帧中 detections 进行匹配（IOU 匹配）
没有 ReID 等深度学习特征（加入 ReId 会让匹配更精确一些）



