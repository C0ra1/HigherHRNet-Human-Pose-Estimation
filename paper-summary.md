# Deep High-Resolution Representation Learning for Human Pose Estimation (CVPR 2019)论文阅读

高分辨率网络解决人体姿态估计问题

论文：[Deep High-Resolution Representation Learning for Human Pose Estimation - CVPR2019](https://arxiv.org/abs/1902.09212 "1")

作者：Ke Sun1, Bin Xiao, Dong Liu, Jingdong Wang

团队：University of Science and Technology of China, Microsoft Research Asia

代码：
[Github - deep-high-resolution-net.pytorch](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch "Github - deep-high-resolution-net.pytorch")

**项目主页：**Projects - [Deep High-Resolution Representation Learning](https://jingdongwang2017.github.io/Projects/HRNet/index.html "Deep High-Resolution Representation Learning") 

#### HRNets

	现有的方法从高到低的分辨率网络产生的低分辨率中回复高分率表示。
	
	HRNets 在网路整个过程中维持或学习高分辨率的特征，渐进的增加低分辨率的的分支，
	并且通过特征交换模块，多次进行高分辨率和低分辨率的特征交换，从而使低分辨率的分支学到足够丰富的高分辨率特征。
	
	区别：高分辨率特征和低分辨率特征并行连接。

### 一、论文背景
>姿态识别主流的方法通常有两个思路：
##### 1.Top-Down方法
	将人体检测和关键点检测分离，在图像上首先运行一个人体检测器，找到所有的人体实例，对每个人体子图再使用关键点检测，
	这类方法往往极其慢，但姿态估计准确度较高。
>**目前的主流是cpn，hourglass，cpm，alpha pose等。**

##### 2.Bottom-Up方法
	先检测图像中人体部件，然后将图像中多人人体的部件分别分配到不同的人体实例上，因此这类方法在测试推断的时候往往更快速,模型Size更小。
	但因为没能更细致的对人体关键点建模，所以往往准确度稍低。典型就是COCO2016年人体关键点检测冠军open pose

>此文章主要介绍了top-down方法中的single-person Pose Estimation网络的设计上
给定一张image输入cnn网络中，得到一个高分辨率特征图heatmap，求最大点key point的位置。
这其中最重要的是如何设计一个网络得到高分辨率的表达特征。

##### 在human pose estimation和高分辨率表达上有名的工作：
###### 1.Stacked Hourglass算法
![hourglass](https://c0ra1.site/wp-content/uploads/2020/03/hourglass.png)

	目前姿态研究的基础网络，具有bottom-up和top-down二者特性。
	Hourglass模块是对称结构，bottom-up过程将图片从高分辨率降到低分辨率，top-down过程将图片从低分辨率升到高分辨率。
	这种网络结构包含了许多pooling和upsampling的步骤，pooling用于提取特征减少冗余，upsampling可以结合多个分辨率的特征。

###### 2.CPN(Cascaded PyramidNetwork)	

>算法包含两个阶段，GlobalNet 和 RefineNet。

![CPN](https://c0ra1.site/wp-content/uploads/2020/03/cascaded-pyramid-network.png)
	
	GlobalNet 是一个特征金字塔网络，它可以找到所有“简单”的关键点，比如眼、手;重叠的或者不可见的关键点就可能无法准确识别。
	RefineNet 是专门用来处理“难”的关键点的，它会把 GlobalNet 中所有级别的特征表征和一个难关键点的挖掘损失集成到一起。

>为了解决多人姿态预测问题，此算法采用了一个逐步细化的流水线，首先用检测器生成一组边界框，框出图中的人体，然后用级联金字塔网络 CPN在每个人体边界框中定位关键点。

###### 3.Simple Baselines
![simple-baselines](https://c0ra1.site/wp-content/uploads/2020/03/simplebaseline.png)
	通过resnet backbone得到深层的low-Resolution Representation。
	通过3个级联的convolution恢复出一个high- Resolution Representation，它在coco取得了很好的成绩。


###### 三个算法设计原则

	以上三个方法的设计原则都是
	先通过high- Resolution Representation的表达慢慢得到一个low-Resolution Representation，再通过一定的方法，如Hourglass 通过对称的upsampling和convolution恢复出high- Resolution Representation。
	
	CPN也是通过upsampling和Skip Connection得到high- Resolution Representation。
	
	Simple Baselines也是慢慢把Resolution从high- Resolution Representation降到low-Resolution Representation，通过Deconvolution恢复出high- Resolution Representation的表达。

##### 前人方法总结：

- 都是从high resolution到low resolution通过级联方式的网络。
- 都是从low resolution representlations用不同的办法恢复high resolution representlations。
- 在Hourglass，CPN中都通过深层的high level feature和浅层的low level feature融合来增强high resolution representlations。
	

>但作者认为之前的高分辨率都是用过相对低的分辨率中恢复出来的，
因此在本文中作者在设计网络时，保持高分辨率:

### 二、HRNet

>本论文网络结构HRNet(High Resolution Net)

[![高分辨率](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch/raw/master/figures/hrnet.png)](http://https://github.com/leoxiaobin/deep-high-resolution-net.pytorch/raw/master/figures/hrnet.png)


>本文作者在网络过程中保持高分辨率的主干，在过程中不降低分辨率
>
>(1) 增加了有高分辨率到低分辨率的子网来获取更多其他信息
>(2) **交换单元** (Exchange Units) ，这个单元将不同的子网络进行融合：让每一个子网络，都可以从其他子网络获取特征信息。

>特征融合一般采用的是，底层特征表示与高层特征表示的聚合融合；而HRNet 采用的重复性的多尺度融合，基于相同网络深度和相似特征层的低分辨率表示的辅助，来提升高分辨率的表示能力，以使得高分辨率的表示更有助于姿态估计。

##### 主要内容：

> 采用 CNN 网络来预测人体关键点，其主要包括两个衰减特征分辨率的步长卷积(strided conv)网络：主干网络(main body)输出与输入特征图相同分辨率的特征图(feature maps)；一个回归器，用于估计 heatmaps 中关键点的位置，并转换回原来的全分辨率. HRNet 关注于主干网络的设计。

#### 2.1. 序列化的多分辨率子网络 

>采用序列地连接各高分辨率到低分辨率的子网络进行构建的，其中，每个 stage 的每个子网络都是包含多个卷积序列，且邻近的子网络间，会有下采样层，以将特征分辨率减半。

- 记 Nsr 为第 s 个 stage的子网络，r 为分辨率索引(特征图分辨率是第一个子网络分辨率的 1/2^(r−1)).
- 包含 S 个 stages 高分辨率到低分辨率的网络可以表示为(假设 S=4)：
>N11→N22→N33→N44

#### 2.2. 并行化多分辨率子网络

> 采用一个高分辨率的子网络作为第一个 stage，然后一个接一个的逐渐添加高分辨率到低分辨率的子网络，并并行地连接各个多分辨率子网络。

> 因此，后一个 stage 的并行子网络的分辨率特征包含了其前一 stage 的分辨率特征以及额外的低分辨率的特征.

![parallel](https://c0ra1.site/wp-content/uploads/2020/03/parallel-net.png)

#### 2.3. 交换单元 (Exchange Units) 

>高、中、低分辨率的信息过程

![exchange](https://c0ra1.site/wp-content/uploads/2020/03/exchange-unit.png)



**重复性多尺度融合**

>这个单元将不同的子网络进行融合，进行信息交换。

>让每一个子网络，都可以从其他子网络获取特征信息。

![change-rule](https://c0ra1.site/wp-content/uploads/2020/03/change-rule.png)

>它是设计通过差值upsampling加1*1 的卷积融合到高分辨率
本文还设计一个模块通过步长等于2，慢慢把分辨率降到和低分辨率一样的情况，再进行高分辨率特征和低分辨率特征的融合。
通过这种设计高分辨率既得到了global的信息的一个增强，同时低分辨率因为得到高分辨率的一些细节信息而得到增强。

>HRNet模型与之前主流方法思路上有很大的不同。在HRNet之前，2D人体姿态估计算法是采用（Hourglass/CPN/Simple Baseline/MSPN等）将高分辨率特征图下采样至低分辨率，再从低分辨率特征图恢复至高分辨率的思路（单次或重复多次），以此过程实现了多尺度特征提取的一个过程。

>HRNet的主要特点是在整个过程中特征图（Feature Map）始终保持高分辨率，低分辨率和高分辨率是并行设计的，他们是在相同的level上。
低分辨率和高分辨率的融合，他们在特征层面基本上是相似的或者相同的。

#### 2.4.Heatmap估计

>只采用最后一个 **exchange unit** 输出的高分辨率特征表示，来回归 heatmaps。 

>loss 函数采用的是均方差误差(mean squared error)，对预测的 heatmaps 和 groundtruth heatmaps 进行计算。

>groundtruth heatmaps 是通过以关键点 groundtruth 坐标(x, y) 为中心，采用 1 pixel 标准差的 2D Gaussian 生成。

#### 2.5. 网络实例化(Network instantiation)

>采用了 ResNet 的设计原则，将网络深度(depth) 分发到每个 stage，将网络通道数(channels) 分发到每个分辨率。

>HRNet 的主干网络，包含有 4 个并行子网络的4个stages，其分辨率逐渐衰减一半，对应的网络宽度(通道数)增加2倍.

	1）整体流程：4 stages
	input–>>stages1(conv1–>bn1–>conv2–>bn2–>layer1)–>>stages2(transition1–>stage2)–>>stages3(transiton2–>stage3)–>>stages4(transiton3–>stage4)–>>final_layer
	2）stages1与resnet50第一个res2相同，包括4个bottleneck，四个残差单元(residual units)
	3）stages2，3，4分别拥有1，4，3个exchange blocks；每个exchange blocks也包含4个bottleneck构成
	4）从上到下，每个stages分辨率减半，通道增倍，文中提到HRNet-W32和HRNet-W48，指的是这些stage的通道数不同，但结构相同
	5）从整体上看，与resnet50极为相似，但多了些过渡单元transition和并行子网络，以及exchange需要的操作

- 实现中，采用了一个小网络 **HRNet-W32** 和一个大网络 **HRNet-W48**.。

- 其中，32 和 48 分别表示在最后三个 stages 中高分辨率子网络的宽度(C)。

- **HRNet-32** 的其它三个并行子网络的宽度分别是 64，128，256。

-  **HRNet-48** 的其它三个并行子网络的宽度分别为 96， 192， 384。

### 三、experiments

#### 3.1. COCO Keypoints Detection

##### 数据集

> COCO 数据集包含超过 200000 张图片，250000 个标注的人体实例，每个人体包含 17 个关键点. 模型训练采用的 COCO train2017 数据集共有 57K 张图片，150K 人体实例. 在 val2017 和 test-dev2017 数据集上验证模型，其分别包含 5000 张图片和 20K 张图片.

##### 评价度量
>COCO 提供的标注评价度量标注是基于 **Object Keypoint Similarity(OKS)**

![OKS](https://c0ra1.site/wp-content/uploads/2020/03/oks.png)

> 采用标准的精度(precision) 和召回分数(recall scores)

> test结果：

![test-result](https://c0ra1.site/wp-content/uploads/2020/03/test-result-e1585502795629.png)

#### 3.2. MPII Human Pose Estimation

> **数据集**：MPII 人体姿态数据集包含 25K 张图片，40K 个标注实例，其中 12K 用于测试。
>
> **训练**：训练策略与 COCO 数据集上的训练一致，除了网络输入图片尺寸裁剪为 256×256，以便于对比。
>
> **测试**：测试策略也基本与 COCO 数据集上的一致，除了采用提供的人体框，而不再进行人体框检测
>
> **评价度量**：采用标准方式，PCKh(head-normalized probability of correct keypoint) score. 如果关节点位置位于 groundtruth 位置的 αl 个像素位置内，即认为关节点估计正确. 
>
> 其中，α 为常数，l 为 head 尺寸，其对应于 groundtruth 的头部边界框对角线长度的 60%. 一般采用 PCKh@0.5(α=0.5)。
