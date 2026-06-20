---
title: Transformer_code
description: ""
published: 2026-06-17
slug: transformer_code
date: 2026-05-11T21:40:48+08:00
categories:
  - AI Infra
series:
  - Transformer 代码实现
tags: []
draft: true
---
## pytorch 常用模块

### 1. torch.nn 模块

如果你把 PyTorch 看作是一个庞大的机械加工厂，那么 `torch` 基础库（比如 `torch.matmul`, `torch.sum`）就像是扳手、锤子这样的基础数学工具。

而 **`torch.nn` (Neural Network)**，则是这个工厂里高度自动化的“数控机床”和“预制零件库”。

`nn` 模块的核心哲学只有一个：**封装状态（State）与行为（Behavior）**。它让你不再需要手动管理每一个权重矩阵的内存和偏置，而是像搭积木一样快速构建复杂的深度学习模型。

我们可以把 `torch.nn` 拆解为**四大支柱**来理解：

---

#### 1.1 骨架与容器 (Containers)

这是 `torch.nn` 的地基，负责组织和管理网络结构。

- **`nn.Module`**：万物之基。我们在前面详细聊过，它是所有层和模型的父类。只要继承了它，你的类就拥有了参数追踪、设备转移（`.to('cuda')`）等超能力。
    
- **`nn.Sequential`**：自动挡容器。如果你有一个像“糖葫芦”一样一条路走到黑的网络结构，用它串起来，连 `forward` 函数都不用自己写。
    
- **`nn.Parameter`**：身份标签。把普通的 Tensor 用它包一层，PyTorch 就会知道：“哦，这个张量是需要梯度更新的模型权重”，并自动把它加入到 `model.parameters()` 中。
    
- **`nn.ModuleList` / `nn.ModuleDict`**：高级收纳盒。当你想在 `__init__` 里用 `for` 循环创建 12 层 Transformer Block 时，必须把它们装进 `nn.ModuleList` 里，否则 PyTorch 无法识别它们。
    

#### 1.2 核心计算层 (Layers)

这里面装着各种自带权重（$W$ 和 $b$）的“预制零件”。

- **线性层 (`nn.Linear`)**：我们之前拆解过的空间投影仪，做全连接操作。
    
- **视觉层 (`nn.Conv1d/2d/3d`)**：卷积神经网络的核心，用来提取局部特征（比如图像边缘）。
    
- **序列层 (`nn.RNN`, `nn.LSTM`, `nn.GRU`)**：处理时间序列和文本的经典循环层（在 Transformer 出现前的 NLP 霸主）。
    
- **注意力层 (`nn.MultiheadAttention`, `nn.Transformer`)**：封装好了整个多头注意力机制。
    

#### 1.3 非线性与归一化 (Activations & Normalization)

只有线性运算的话，叠 100 层等价于叠 1 层。这些层负责为模型注入“灵魂”和维持“稳定”。

- **激活函数 (`nn.ReLU`, `nn.GELU`, `nn.Sigmoid`)**：给网络加入非线性表达能力。比如 GPT 和 BERT 大量使用的就是 `GELU`。
    
- **归一化层 (`nn.BatchNorm2d`, `nn.LayerNorm`)**：控制数据方差。图像常用 BatchNorm，而 Transformer 这种序列模型死磕 LayerNorm。
    
- **正则化层 (`nn.Dropout`)**：我们聊过的防过拟合神器，训练时随机让神经元“断电”。
    

#### 1.4 判决者：损失函数 (Loss Functions)

模型算出了结果，谁来评判它对不对？这就是损失函数的工作，它输出一个误差标量，驱动模型进行反向传播。

- **`nn.CrossEntropyLoss`**：分类任务之王（比如预测下一个单词是什么，或者区分猫和狗）。它内部其实把 `Softmax` 和 `LogLikelihood` 封装在一起了，算得又快又稳。
    
- **`nn.MSELoss`**：回归任务必备，计算均方误差（比如预测房价是多少）。
    

---

#### 1.5 💡 核心代码演示

下面是一段标准的 PyTorch 代码，展示了 `torch.nn` 里的这些组件是如何默契配合的：

```python
import torch
import torch.nn as nn

# 1. 继承 nn.Module 容器
class SimpleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 2. 组合 Layers、Normalization 和 Activations
        self.net = nn.Sequential(
            nn.Linear(768, 256),     # 降维投影
            nn.LayerNorm(256),       # 维持方差稳定
            nn.ReLU(),               # 非线性激活
            nn.Dropout(p=0.1),       # 防止过拟合
            nn.Linear(256, 10)       # 映射到 10 个类别
        )

    def forward(self, x):
        # 数据的正向流动极其简单
        return self.net(x)

# 实例化模型
model = SimpleClassifier()

# 3. 准备评判者 (Loss Function)
criterion = nn.CrossEntropyLoss()

# 模拟输入 (Batch=32, 维度=768) 和真实标签 (类别 0-9)
dummy_x = torch.randn(32, 768)
dummy_labels = torch.randint(0, 10, (32,))

# 正向传播
predictions = model(dummy_x)

# 计算损失
loss = criterion(predictions, dummy_labels)
print(f"当前模型的误差 (Loss): {loss.item():.4f}")
```

---
### 2 view 函数

在 PyTorch 中，`view` 函数是用来重塑张量形状（Reshape）的绝对主力。

如果把 PyTorch 的底层内存想象成一个**大仓库**，里面的数据（数字）就是按顺序排列的一排排**集装箱**。`view` 函数的伟大之处在于：**它不搬运任何一个集装箱，它只是换了一副“眼镜”，让你用不同的维度和排版去读取这些集装箱。**

我们可以通过四个最核心的实战维度来彻底搞懂它：

---

#### 2.1 基础魔法：改变观察视角

假设我们有一根长度为 12 的一维数据（12 个集装箱排成一列）。

```python
import torch

# 生成 0 到 11 的一维张量
a = torch.arange(12) 
print(a.shape) # 输出: torch.Size([12])
# 内容: tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
```

现在，你想把它变成一个矩阵。由于 $3 \times 4 = 12$，$2 \times 6 = 12$，你可以随意 `view` 它：

```python
# 变成 3 行 4 列的矩阵
b = a.view(3, 4)
print(b)
'''
tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])
'''

# 变成 2 行 6 列的矩阵
c = a.view(2, 6)
```

**核心法则**：`view` 前后的总元素个数（Total Elements）必须**绝对相等**，否则会直接报错。

---

#### 2.2 高级魔法：`-1` 的自动推导

在 `view` 函数中，`-1` 相当于告诉 PyTorch：**“这个维度的数字我懒得算了，你帮我用总数除以其他维度求出来吧。”**

这在处理 Batch Size 未知，或者序列长度不定时极其好用。

```python
x = torch.arange(24) # 总共 24 个元素

# 我只知道我要 3 个 Batch，每个 Batch 2 行，剩下的列数你自己算
# 计算逻辑：24 / (3 * 2) = 4
y = x.view(3, 2, -1) 
print(y.shape) # 输出: torch.Size([3, 2, 4])

# 经典应用：展平操作 (Flatten)
# 把一张 [Batch, Channel, Height, Width] 的图像展平送入 Linear 层
image = torch.randn(32, 3, 224, 224)
# 保留 Batch 维度，剩下的全部拉平成一维
flat_image = image.view(32, -1) 
print(flat_image.shape) # 输出: torch.Size([32, 150528])
```

_注：一个 `view` 中最多只能出现一个 `-1`，否则 PyTorch 解不出方程。_

---

#### 2.3 底层机制：共享内存（极速且危险）

`view` 函数运行极快，**因为它根本没有在内存中复制数据**。它只是修改了张量的“步长（Stride）”等元信息。

这意味着：**新张量和老张量共享同一块物理内存。** 你修改了新张量，老张量也会跟着变！

```python
a = torch.arange(4)
b = a.view(2, 2)

b[0, 0] = 999 # 修改 b 的第一个元素

print(a) # 你会惊讶地发现，a 的第一个元素也变成了 999！
# 输出: tensor([999,   1,   2,   3])
```

---

#### 2.4 最大的坑：`view` vs `reshape`

这是面试中最常考的陷阱，也是初学者最容易报 `RuntimeError: view size is not compatible with input tensor's size and stride` 错误的地方。

- **`view` 的傲娇**：它要求原始张量在内存中的存储必须是**绝对连续的（Contiguous）**。如果你对张量做过 `transpose`（转置）、`permute`（维度换位）等操作，张量在内存里的顺序就乱了。这时候直接调用 `view` 就会报错。
    
- **解决办法**：
    
    1. 先呼叫整理工：`a.transpose(0,1).contiguous().view(...)`
        
    2. **直接使用 `reshape`**：`a.transpose(0,1).reshape(...)`
        

**`reshape` 是什么？**

`reshape` 是 PyTorch 后来引入的更智能的函数。如果你对一个不连续的张量调用 `reshape`，它会在后台默默地帮你开辟一块新内存把数据排好（等价于 `contiguous().view()`）；如果内存本来就是连续的，它就直接做 `view` 的操作。

---
### 3. transpose 函数

**它的作用**：交换张量的两个维度。比如把矩阵的行和列互换（转置）。

**底层揭秘**： `transpose` 极其“偷懒”。当你调用 `a.transpose(0, 1)` 时，它**绝对不会去搬动底层内存中的任何一个数字**。 它仅仅是修改了张量头部信息中的 **Shape（形状）** 和 **Stride（步长）**。

- **什么是步长 (Stride)？** 步长就是“为了在某个维度上移动一步，我在物理内存里需要跳过多少个格子”。
    
- 原本在行上移动一步要跳过 3 个格子，列上移动一步跳过 1 个格子。`transpose` 只是把这两个规则对调了一下。
    

**代价（副作用）**： 因为物理内存没动，但读取顺序变了，这就导致按照新的逻辑去读数据时，内存地址是**跳跃的、不连续的**。 **结论**：`transpose` 之后，张量通常会变成**非连续状态 (Non-contiguous)**。一旦非连续，脾气暴躁的 `view` 函数就会拒绝为它服务。

---

### 4. `contiguous`：强迫症的“整理工”（内存重排）

**它的作用**：检查张量在内存中是不是连续的。如果是，直接返回它自己；如果不是，就**强行开辟一块新内存**，把数据按顺序排好。

**底层揭秘**： 它是一个“实干家”。当你对一个经过 `transpose` 的张量调用 `.contiguous()` 时：

1. 它会向 GPU/CPU 申请一块全新的连续内存空间。
    
2. 它会按照当前的逻辑顺序，把老内存里的数字一个一个**拷贝 (Copy)** 到新内存里。
    
3. 更新头部信息，Stride 恢复为最标准的连续步长。
    

**代价（副作用）**： 因为涉及物理数据的深拷贝，它会**消耗额外的时间和显存空间**。

**结论**：它是平息 `view` 函数怒火的唯一解药。标准连招是：`a.transpose(0, 1).contiguous().view(...)`。

---

### 3. `reshape`：高情商的“智能经理”（智能重塑）

**它的作用**：改变张量的形状。只要元素总数对得上，你想怎么捏就怎么捏。

**底层揭秘**： `reshape` 本身不干底层的脏活累活，它是一个智能的“调度员”。它的运行逻辑是一个完美的 `if-else` 分支：

- **如果** 当前张量的底层内存是连续的：它就直接喊 `view` 来干活。此时**不拷贝数据**，极速完成。
    
- **如果** 当前张量的底层内存是不连续的（比如刚被 `transpose` 过）：它就先喊 `contiguous` 去开辟新内存把数据排整齐，然后再喊 `view` 来改变形状。此时**会拷贝数据**。
    

**结论**：`reshape` = 智能版的 `view`。日常写普通逻辑时，无脑用 `reshape` 可以避开 99% 的形状报错。

---

### 🥷 torch.nn.Dropout：训练时的“蒙眼特训”，测试时的“全军出击”

#### 1. 核心定义与基础语法

`nn.Dropout` 是一种正则化（Regularization）层，专门用来防止模型过拟合（Overfitting）。

- **语法**：`dropout = nn.Dropout(p=0.5)`
    
- **核心参数 `p`**：这是**失活概率（Probability）**。`p=0.5` 意味着在每次前向传播时，张量中大约有 50% 的元素会被残忍地变成 `0`。
    

#### 2. 物理直觉：为什么要“自断双臂”？（防过拟合的本质）

想象一个 100 人的开发团队（神经网络层）要完成一个大项目（拟合数据）。

- **没有 Dropout 时**：团队里有几个“全栈卷王”（权重特别大的神经元），遇到问题总是他们几个上。久而久之，剩下 90 个人就在旁边划水。一旦这几个卷王请假（遇到没见过的新数据），整个项目直接瘫痪。这就是**过拟合（死记硬背）**。
    
- **加上 Dropout(p=0.5) 后**：教官规定，每天随机抽 50 个人关进小黑屋（变成 0），剩下 50 个人必须硬着头皮把活干完。
    
- **最终结果**：在这个残酷的特训下，团队里**每一个人都被逼成了全栈工程师**。神经元之间不再产生严重的相互依赖（共适应性），整个网络变得极度强壮和鲁棒。
    

#### 3. 核心机制：Train 模式 vs Eval 模式（面试必考）

Dropout 是一个极其“双标”的函数，它在训练和测试时的表现完全不一样！这就引出了 PyTorch 中非常重要的两个模式切换函数：`model.train()` 和 `model.eval()`。

##### 🥊 训练阶段 (`model.train()`)

- **动作**：随机将输入张量中的一部分元素置为 `0`（概率为 $p$）。
    
- **⚠️ 隐藏的魔法（Inverted Dropout）**：为了保证输出的**数学期望（总和/平均值）不变**，PyTorch 会在底层悄悄把那些**幸存下来**的元素，放大 **$\frac{1}{1-p}$** 倍！
    
    > **举个例子**：假设输入全是 `1`，`p=0.5`。
    > 
    > 有一半的人倒下了变成了 `0`，如果剩下的不放大，输出的总能量就只剩一半了。
    > 
    > 所以，剩下的那一半人必须**每人打两份工（放大 $\frac{1}{0.5} = 2$ 倍）**，变成了 `2`。这样总能量才保持不变。
    

##### 🕊️ 评估/测试阶段 (`model.eval()`)

- **动作**：**教官休假，Dropout 直接罢工！** 所有的神经元全部保留，原样输出（相当于 `p=0`）。
    
- **为什么？** 因为在真刀真枪的测试/推理环节，我们需要整个团队拿出 100% 的全部实力，绝对不能再搞“随机失活”这种特训了。
    

#### 4. Transformer 中的实战位置

Dropout 出现在 Softmax 之后：

```python
attns = torch.softmax(scores, dim=-1)
attns = self.dropout(attns)
```

- **它的作用**：随机把一部分“注意力概率”捏碎成 `0`。
    
- **业务意义**：强迫模型“不要总是死死盯着那几个高分词汇”。哪怕那个词再重要，今天我也要蒙住你的眼睛，逼你去看看周围那些低分的词，从更广泛的上下文中提取线索。
    

除此之外，在 Transformer 中，你还会在以下两个地方频繁看到 Dropout 的身影：

1. **多头注意力结束，做完残差连接（Add）之前**。
    
2. **前馈神经网络（FFN）的两层线性映射之间**。

#### 💡 互动时间：Dropout “能量守恒”模拟器

很多人（包括面试官）都容易忘记 Dropout 训练时那个 $\frac{1}{1-p}$ 的放大机制。为了让你彻底记住这个底层的数学魔术，我做了一个交互式的 **PyTorch Dropout 模拟器**。

你可以拖动失活概率 $p$，并切换“训练/评估”模式，亲眼看看幸存的神经元是如何被**强行放大**的：

> **你的终极笔记口诀：**
> 
> 1. **目的**：打断神经元之间的共适应，逼迫每一个节点独立提取特征，防止死记硬背（过拟合）。
>     
> 2. **训练时 (Train)**：随机杀掉 $p$ 的节点，剩下的节点放大 $\frac{1}{1-p}$ 倍（保证期望值守恒）。
>     
> 3. **测试时 (Eval)**：原封不动，全部输出。
>

---
### 🔄 torch.transpose：最快但最“危险”的维度互换术

#### 1. 核心定义与基础语法

`transpose` 的字面意思是“转置”。在 PyTorch 中，它的核心功能非常专一：**精确指定并交换张量（Tensor）的两个维度**。

- **语法**：`tensor.transpose(dim0, dim1)`
    
- **规则**：`dim0` 和 `dim1` 是你想要互换的两个维度的索引（可以正数也可以负数，互换顺序不影响结果，即 `transpose(1, 2)` 和 `transpose(2, 1)` 等价）。
    

```python
import torch

# 创建一个 3D 张量：[Batch=32, Seq_Len=10, Feature=512]
x = torch.randn(32, 10, 512)

# 需求：把句子长度 (维 1) 和 特征维度 (维 2) 互换
# 此时 Batch (维 0) 保持不动
y = x.transpose(1, 2) 
# 或者用负数索引：x.transpose(-2, -1)

print(y.shape) # 输出: torch.Size([32, 512, 10])
```

#### 2. 底层揭秘：不搬数据的“换镜片”魔法

这是 `transpose` 最核心的考点。它是一个运行速度为 **$O(1)$**（极速）的操作。

- **为什么这么快？**
    
    当你调用 `transpose` 时，PyTorch **绝对不会在物理内存中搬运任何一个字节的数据**。底层内存（一维数组）的排列顺序纹丝不动。
    
- **它到底干了什么？**
    
    它仅仅是修改了张量头部的元数据（Metadata）：
    
    1. **Shape（形状）**：把两个维度的大小对调。
        
    2. **Stride（步长）**：把在这两个维度上移动所需的“跨步数”对调。
        
- **致命副作用（划重点）**：
    
    因为物理数据没变，但读取顺序变了，导致张量在内存中的读取逻辑变得**跳跃**。这会使张量进入 **非连续状态 (Non-contiguous)**。一旦非连续，后续如果强行调用 `.view()` 就会直接报错！
    
    > **标准连招补救**：`x.transpose(1, 2).contiguous().view(...)`
    

#### 3. 工业界两大经典实战场景

#### 场景 A：Transformer 中的多头并行（NLP 必备）

这正是我们在解析 Multi-Head Attention 时遇到的神仙操作。为了让 8 个头能同时使用批量矩阵乘法（BMM），必须把“头数”这个维度提到前面来充当 for 循环。

```python
# 切分多头后：[Batch, Seq_Len, Num_Heads, d_k]
Q = Q.view(32, 10, 8, 64)

# 乾坤大挪移：交换 Seq_Len(维1) 和 Num_Heads(维2)
Q = Q.transpose(1, 2)

# 最终形状：[Batch, Num_Heads, Seq_Len, d_k] -> [32, 8, 10, 64]
# 完美满足接下来 [10, 64] 和 [64, 10] 的点积相乘！
```

#### 场景 B：图像维度的转换（CV 必备）

在计算机视觉中，用 OpenCV/PIL 读进来的图片格式通常是 `[高, 宽, 通道数]` (H, W, C)。但 PyTorch 的卷积层 `nn.Conv2d` 死磕的要求是 `[通道数, 高, 宽]` (C, H, W)。

```python
image = torch.randn(224, 224, 3) # [H, W, C]

# 第一步：通道数(2)换到宽(1)的前面 -> [224, 3, 224]
img_tensor = image.transpose(1, 2) 

# 第二步：通道数(1)再换到高(0)的前面 -> [3, 224, 224]
img_tensor = img_tensor.transpose(0, 1)
```

#### 4. 易混淆辨析：`transpose` vs `permute` vs `.T`

既然提到了 `transpose`，面试和实战中绝对绕不开它的两个兄弟。

|**函数**|**作用域**|**特点与代码对比**|
|---|---|---|
|**`transpose`**|**只能交换 2 个维度**|像手术刀，精确但一次只能切一刀。如果是多维重排，得像上面CV例子一样写好几次。|
|**`permute`**|**可以同时重排所有维度**|像魔法阵，直接按你期望的索引顺序排好。CV维度的转换通常更爱用它：<br><br>  <br><br>`image.permute(2, 0, 1)` 一步到位实现 [C, H, W]。|
|**`.T`**|**快捷反转**|专为 2D 矩阵准备的语法糖，等价于 `transpose(0, 1)`。如果是多维，它会直接把所有维度倒序，容易失控，**在多维张量中慎用**！|

#### 💡 终极一句话口诀

> **“`transpose` 换维快如电，物理内存全没变；若想接上 `view` 变形，先用 `contiguous` 把坑填。”**

---

### torch.unsqueeze：无中生有的“维度扩充术”

#### 1. 核心定义与基础语法

`unsqueeze` 的字面意思是“取消挤压/展开”。它的唯一作用是：**在张量的指定位置，强行插入一个长度为 1 的新维度。**

- **语法**：`tensor.unsqueeze(dim)`
    
- **规则**：`dim` 是你想要插入新维度的索引位置（范围从 `-input.dim() - 1` 到 `input.dim()`）。
    

```python
import torch

# 原始数据：一维张量（比如 3 个数字排成一排）
x = torch.tensor([1, 2, 3]) 
print(x.shape) # 输出: torch.Size([3])

# 魔法 1：在最前面（索引 0）插入维度
y = x.unsqueeze(0) 
print(y.shape) # 输出: torch.Size([1, 3])
print(y)       # 输出: tensor([[1, 2, 3]]) -> 变成了一行三列的矩阵

# 魔法 2：在最后面（索引 1 或 -1）插入维度
z = x.unsqueeze(1)
print(z.shape) # 输出: torch.Size([3, 1])
print(z)       
''' 输出: 
tensor([[1],
        [2],
        [3]]) -> 变成了三行一列的矩阵
'''
```

#### 2. 直观比喻：“包装盒理论”

我们可以用**包装盒**来完美理解 `unsqueeze`：

底层的数据（苹果）数量永远没有变，`unsqueeze` 只是在加包装。

- 原始 `[3]`：桌子上散落着 3 个苹果。
    
- `.unsqueeze(0)` 变成 `[1, 3]`：拿 **1 个大长条盒子**，把这 3 个苹果装进去。（整体看是 1 个盒子，里面有 3 个苹果）。
    
- `.unsqueeze(1)` 变成 `[3, 1]`：拿 **3 个小方盒子**，每个盒子里装 1 个苹果。（整体看是 3 个盒子，每个只有 1 个苹果）。
    

#### 3. 底层揭秘：极速的元数据修改

和 `view`、`transpose` 一模一样，`unsqueeze` 也是一个 **$O(1)$ 的极速操作**。

它**绝对不会**在内存里去复制或搬运任何底层数据。它仅仅是在张量的“头部信息（Shape）”里插了一个 `1`，并调整了相应的步长（Stride）。因此，它极度高效，不消耗额外显存。

#### 4. 工业界两大经典实战场景

##### 场景 A：单样本推理时的“强行凑 Batch”（CV/NLP 必备）

深度学习模型（如 `nn.Conv2d` 或 `nn.Linear`）在训练时习惯了吃“批次”数据。比如图像模型的输入标准是 `[Batch, Channel, Height, Width]`。

但当你在生产环境中测试**单张图片**时，图片读取进来的形状往往是 `[C, H, W]`，少了一个 Batch 维度，直接送进模型会报错。

```python
image = torch.randn(3, 224, 224) # [C, H, W]

# 强行在最前面插入一个 Batch 维度
batch_image = image.unsqueeze(0) 

# 形状变成 [1, 3, 224, 224]，模型开心接收！
output = model(batch_image)
```

##### 场景 B：触发自动广播（Broadcasting）机制的“垫脚石”

这正是我们在上一节 `attn_mask` 代码里遇到的场景。

当两个维度不匹配的张量要做加减乘除时，我们需要用 `unsqueeze` 把它们“垫”到相同的维度数量，告诉 PyTorch 哪些维度需要对齐，哪些维度的 `1` 可以被复印（广播）放大。

```python
# 例如：给 [32, 10, 10] 的矩阵，强行在第 1 维塞个 1
# 变成 [32, 1, 10, 10]，为后续的 repeat 或广播操作预留出 num_heads 的位置。
mask = mask.unsqueeze(1) 
```

#### 5. 它的反义词：`squeeze()`

既然能装盒子，就能拆盒子。

`tensor.squeeze()` 的作用是**删除所有长度为 1 的维度**（把毫无意义的单层包装盒拆掉）。也可以指定删除某个特定的维度：`tensor.squeeze(dim)`。

```python
a = torch.randn(1, 3, 1, 224)
b = a.squeeze() 
print(b.shape) # 输出: torch.Size([3, 224]) ，两个 1 都被捏碎了！
```

#### 💡 终极笔记口诀

> **关于 `unsqueeze` 的核心记忆点：**
> 
> 1. **功能**：在哪填数字，就在哪加个 `1` 的维度（无中生有）。
>     
> 2. **数据**：内存不移，数据不加，纯粹加了层“逻辑包装盒”。
>     
> 3. **用法**：常用于**单样本凑 Batch**，或者为了**触发广播机制**做维度对齐的垫脚石。
>

---
### 🖨️ torch.repeat：真正的“物理复印机”

#### 1. 核心定义与基础语法

`repeat` 的作用非常简单粗暴：**沿着指定的维度，将张量里的数据物理复制指定的次数。**

- **语法**：`tensor.repeat(*sizes)`
    
- **⚠️ 致命易错点**：传入的参数表示的是**在这个维度上要复印几份**，而**绝对不是**期望的目标形状（Shape）！
    

```python
import torch

# 原始数据：形状为 [2, 3] 的矩阵
x = torch.tensor([[1, 2, 3], 
                  [4, 5, 6]]) 

# 魔法 1：在行（维0）复印 2 份，列（维1）复印 1 份（不变）
y = x.repeat(2, 1) 
print(y.shape) # 输出: torch.Size([4, 3])  (因为 2*2=4，3*1=3)
print(y)
'''
tensor([[1, 2, 3],
        [4, 5, 6],
        [1, 2, 3],
        [4, 5, 6]])
'''

# 魔法 2：高阶复印（自动补充维度）
# 如果传入的参数个数大于张量本身的维数，PyTorch 会先自动在前面加上 unsqueeze(0)
z = x.repeat(2, 2, 1) # x 先变成 [1, 2, 3]，然后再复印
print(z.shape) # 输出: torch.Size([2, 4, 3])
```

#### 2. 底层揭秘：吃显存的“老实人”

与我们之前学过的 `view`、`transpose`、`unsqueeze` 完全不同！ `repeat` 是一个**重资产**操作。

当你调用 `repeat` 时，PyTorch 会：

1. **去显存里开辟一块巨大的全新空间**。
    
2. 像老黄牛一样，把原来的数据一个字节一个字节地真实拷贝（Copy）进新空间里。
    
3. 返回的新张量，在内存中是**绝对连续的（Contiguous）**。
    

> **代价**：如果你的张量很大，`repeat` 会瞬间吃掉大量的显存和计算时间。

#### 3. 它的“省钱双胞胎”：`expand`（面试必考）

既然 `repeat` 这么消耗显存，那有没有不消耗显存的平替呢？有，它就是 `expand`！

- **`expand` 的定义**：它是“逻辑上的复印机”。它**不占用新内存，不物理拷贝数据**，仅仅通过把特定维度的步长（Stride）设置为 0，让 PyTorch 在读取时产生“复印了”的幻觉。（这也就是我们在前面说到的“自动广播机制”的底层原理）。
    
- **`expand` 的限制**：它只能对长度为 `1` 的维度进行复印。
    
- **参数的区别（极度容易搞混）**：
    
    - `repeat(2, 3)`：行数乘以2，列数乘以3。
        
    - `expand(2, 3)`：不管原来怎样，**强制把最终形状变成 [2, 3]**（前提是符合广播规则）。
        

```python
a = torch.tensor([[1, 2, 3]]) # 形状 [1, 3]

# 用 repeat：真复印，占用 2x3 个数字的内存
b = a.repeat(2, 1) 
print(b.shape) # [2, 3]

# 用 expand：假复印，底层还是只有 3 个数字的内存
c = a.expand(2, 3) 
print(c.shape) # [2, 3]

# ⚠️ 危险警告：修改 expand 后的张量，会影响原张量！
c[0, 0] = 999 
print(a) # a 的第一个元素也变成了 999！
```

#### 4. 工业界实战指南：到底用哪个？

既然 `expand` 这么省显存，为什么在 Attention 源码里（比如你之前发给我的 `attn_mask.unsqueeze(1).repeat(...)`）还要用 `repeat` 呢？

- **首选 `expand`**：90% 的情况下，如果你只是为了把形状对齐好去做加减乘除（比如广播运算），**无脑用 `expand`**。
    
- **必须用 `repeat` 的场景**：
    
    1. 你需要修改复印后的数据，且**不想影响原数据**。
        
    2. 后续的操作（比如某些特定的 cpp 底层算子、或者 `view`）强硬要求张量必须是**内存物理连续 (Contiguous)** 的。`expand` 出来的张量绝对不是连续的，如果强行用 `contiguous()`，底层依然会发生物理拷贝，那还不如一开始就用 `repeat` 让代码更直观。
        

#### 💡 终极笔记口诀

> **关于 `repeat` 与 `expand`：**
> 
> 1. **`repeat` 是复印倍数**，**`expand` 是目标形状**。
>     
> 2. **`repeat` 是真金白银买新房（真拷贝，耗显存，连续）**。
>     
> 3. **`expand` 是海市蜃楼造幻觉（假拷贝，省显存，非连续）**。
>     
> 4. **能用广播/expand 解决的，绝不轻易用 repeat！**
>

---
### 🎭 Tensor.bool()：一锤定音的“真假判官”

#### 1. 核心定义与基础语法

`.bool()` 的字面意思是“转换为布尔型”。它的作用非常纯粹：**将张量（Tensor）里的所有数值数据，强制转换成 `True` 或 `False` 的逻辑数据。**

- **转换规则**：在 PyTorch 中，遵循编程界的通用铁律：**非零即真**。
    
    - 所有的 `0` 都会变成 `False`。
        
    - 所有的 `1`（或者其他任何非零数字，如 `255`, `-1`）都会变成 `True`。
        

```python
import torch

# 原始掩码张量（通常由 0 和 1 组成）
# 比如 1 代表 PAD（填充词，需要被遮盖），0 代表真实词（不需要遮盖）
mask_data = torch.tensor([[0, 0, 1], 
                          [0, 1, 1]])

# 施加 .bool() 魔法
bool_mask = mask_data.bool()

print(bool_mask)
'''
tensor([[False, False,  True],
        [False,  True,  True]])
'''
```

#### 2. 为什么在这里“非用不可”？（核心考点）

在 Attention 的代码里，我们费了这么大劲把 `attn_mask` 变形为 `[Batch, Heads, L, L]`，紧接着就要用它去遮盖刚刚算出来的 Attention Score 矩阵了。

在 PyTorch 中，执行这个“遮盖”动作的唯一指定函数是 **`masked_fill_(mask, value)`**。

**🚨 PyTorch 的底层死规矩：**

`masked_fill_` 函数的底层 cpp 源码中，有极其严格的类型校验。它规定：**传进来的 `mask` 张量，其数据类型必须绝对是 `torch.bool`（布尔型）。**

如果你传一个普通的 `Float` 张量或者 `Int`（0 和 1）张量进去，程序会当场崩溃，报出深度学习界最经典的新手错误之一：

> `RuntimeError: masked_fill_ only supports boolean masks, but got Byte/Int/Float...`

所以，加上 `.bool()`，就是为了给 `masked_fill_` 递上一张符合规矩的“通行证”。

#### 3. 底层联动：Attention 的“记忆消除术”

为了让你彻底看懂 `.bool()` 的价值，我们必须看看它紧接着参与的下一行代码是什么样的。

通常，它的下一行长这样：

```python
# scores 是 Q 和 K 相乘算出来的打分矩阵
scores = scores.masked_fill_(attn_mask, -1e9)
```

这套连招的物理意义是极其绝妙的：

1. **`.bool()` 定义了规则**：`True` 代表“这个词是违禁词（或者是未来的词），不准看！”，`False` 代表“这个词是正常的，可以看。”
    
2. **`masked_fill_` 执行了抹杀**：它拿着这个布尔表格去对照 `scores` 矩阵。只要看到 `True` 的地方，它就把 `scores` 里的原始打分**残忍地替换成一个极其微小的负数：`-1e9`（负十亿）**。看到 `False` 的地方，保持原分数不动。
    

**为什么要替换成 `-1e9` 这么奇怪的数字？**

因为紧接着就要做 `Softmax` 操作了，要把分数转化为概率。

在数学公式 $Softmax(x_i) = \frac{e^{x_i}}{\sum e^{x_j}}$ 中：

- 当 $x = -1000000000$ 时，$e^{-1000000000}$ 的结果**无限趋近于绝对的 0**。
    
- 这样一来，那些被遮盖掉的词，它们分到的注意力概率权重就变成了 `0`。在最后和 $V$ 矩阵相乘时，它们包含的特征信息就会被彻底丢弃，实现了完美的“物理隔绝”。
    

#### 💡 终极笔记口诀

> **关于 `.bool()` 在 Attention 中的使命：**
> 
> 1. **作用**：把 `0/1` 转化为 `False/True`。
>     
> 2. **目的**：满足 `masked_fill_` 函数严苛的底层数据类型检查。
>     
> 3. **战术地位**：它是“记忆消除术”的判决书。被判 `True` 的位置，将被打入 `-1e9` 的冷宫，最终在 Softmax 的火焰中化为概率 `0`。
>
---
这是一份为你量身定制的 PyTorch `matmul` 函数硬核复盘笔记。

在 PyTorch 的世界里，如果说 `view` 和 `transpose` 是空间魔术师，那么 **`matmul` 就是真正掌控火力的“主战坦克”**。深度学习中 90% 的算力，都消耗在这个函数里。

---

### ⚔️ torch.matmul：八面玲珑的“智能乘法编译器”

#### 1. 核心定义与基础语法

`matmul` 是 Matrix Multiplication（矩阵乘法）的缩写。它的核心功能是执行两个张量的乘法操作。

- **语法**：`torch.matmul(tensor1, tensor2)`
    
- **👑 究极语法糖**：在 python 3.5 之后，你可以直接用 **`@`** 符号来完美等价替代 `matmul`！
    
    > `torch.matmul(A, B)` 等价于 **`A @ B`**（源码中极度常见，极其优雅）。
    

#### 2. 底层揭秘：它为什么叫“智能”编译器？

在线性代数的课本里，矩阵乘法只针对 2D 矩阵。

但 `matmul` 极其聪明，**它会根据你传入张量的维度（1D、2D、3D 甚至 5D），自动切换四套完全不同的底层乘法逻辑！**

我们将这四套逻辑称为 **“matmul 的四重境界”**：

##### 🟢 第一重：1D @ 1D （向量点积）

当两个张量都是一维时，它退化成最简单的**点积（Dot Product）**，也就是对应元素相乘再相加。

- **规则**：`[N] @ [N] -> 标量 (Scalar)`
    

```python
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
print(a @ b) # 输出 32 (1*4 + 2*5 + 3*6)
```

#### 🟡 第二重：2D @ 2D （标准矩阵乘法）

这就是最经典的大学线性代数。

- **规则**：`[M, K] @ [K, N] -> [M, N]` （左边的列数必须等于右边的行数，像拉链一样咬合）。
    

#### 🟠 第三重：1D @ 2D 或 2D @ 1D （降维打击与自动伸缩）

如果一个是一维向量，一个是二维矩阵，PyTorch 会做极其聪明的“临时变形”：

- **例如 `[K] @ [K, N]`**：
    
    PyTorch 会在脑海里悄悄把左边 `unsqueeze` 成 `[1, K]`，算出结果 `[1, N]`，然后再悄悄 `squeeze` 掉那个 `1`，把最终结果变回一维向量 **`[N]`**。
    

#### 🔴 第四重：高维 @ 高维 （批量矩阵乘法 BMM + 广播机制）

**（🔥 这就是你在 Transformer 源码里看到的那一行！）**

这就是我们上一节反复强调的“办公大楼 CEO 规矩”。

- **规则**：**只拿最后两个维度做标准 2D 矩阵乘法，前面的所有维度全部当成 Batch 循环！**
    
- **最牛逼的隐藏技能（广播机制）**：
    
    如果前面代表“房间号”的维度对不上怎么办？它会自动广播（也就是调用类似 `expand` 的假复印逻辑）！
    
    > 比如：`[32, 1, 10, 64] @ [32, 8, 64, 10]`
    > 
    > 左边在第 1 维（头数）上只有一个房间，右边有 8 个房间。`matmul` 会自动把左边的 1 个房间“广播”成 8 个一模一样的房间，然后再进房间算乘法。最后输出完美匹配的 `[32, 8, 10, 10]`！
    

#### 3. 面试/工程必考：`matmul` 的三个同父异母兄弟

在看开源代码时，你经常会看到下面这三个函数，它们到底和 `matmul` 有什么区别？

|**乘法函数**|**性格特点**|**适用场景**|**维度限制**|
|---|---|---|---|
|**`torch.mm`**|**刻板的老教授**|最纯粹的 2D 矩阵乘法。没有任何智能广播，没有魔法。|**只能 `2D @ 2D`**|
|**`torch.bmm`**|**死板的包工头**|专门做批量矩阵乘法的底层算子（Batched MM）。速度极快，但**决不允许广播**，前面代表 Batch 的维度必须完全一致。|**只能 `3D @ 3D`**<br><br>  <br><br>(如 `[B,M,K] @ [B,K,N]`)|
|**`torch.dot`**|**专一的点积工**|只做一维向量的点乘。|**只能 `1D @ 1D`**|
|**`torch.matmul (@)`**|**八面玲珑的 CEO**|**集成了上面三者的所有功能**，外加极度聪明的自动对齐和广播机制。|**无维度限制，万物皆可 `@`**|

> **血泪经验：** 在实现 Transformer 时，早期很多人用 `torch.bmm`，这就导致在使用前必须把 4D 张量（`[Batch, Heads, L, d_k]`）强行 `.view` 压缩成 3D 张量（`[Batch*Heads, L, d_k]`），算完之后再 `.view` 恢复成 4D。
> 
> 后来大家都学聪明了，**直接用 `torch.matmul` 或 `@`**，让 PyTorch 的底层自己去处理那完美的 4D 结构，代码极其优雅！

---

### 💡 终极笔记口诀

> **关于 `matmul` (@) 的四句真言：**
> 
> 1. **一维遇一维**，变成标量飞。
>     
> 2. **二维遇二维**，线代基本规。
>     
> 3. **高维遇高维**，只看最后俩（维度），前面全是 Batch 循环大狂欢。
>     
> 4. **万物皆可用 `@`，告别 `bmm` 和 `mm` 的死板约束！**
>
----

### 🥷 torch.Tensor.masked_fill_：杀伐果断的“记忆消除器”

### 1. 核心定义与基础语法

`masked_fill_` 的作用非常纯粹：**拿着一张布尔型（True/False）的“通缉令”（mask），在你的目标张量里逐个比对。凡是对上号的（为 True 的位置），立刻将其无情地替换成你指定的“惩罚值”（value）。**

- **语法**：`tensor.masked_fill_(mask, value)`
    
- **前提条件**：`mask` 必须是 `torch.bool` 类型，且形状必须能与 `tensor` 对齐（支持广播机制）。
    

```python
import torch

# 1. 假设这是刚算出来的 Attention Score
scores = torch.tensor([[1.2, -0.5, 3.1],
                       [0.8,  2.2, 1.5]])

# 2. 假设这是我们上一节生成的 bool 掩码
# True 代表“这是违禁词，必须遮盖”
mask = torch.tensor([[False, False, True],
                     [False, True,  True]])

# 3. 施加惩罚：把 mask 为 True 的地方，替换成 -1e9
scores.masked_fill_(mask, -1e9)

print(scores)
'''
tensor([[ 1.2000e+00, -5.0000e-01, -1.0000e+09],
        [ 8.0000e-01, -1.0000e+09, -1.0000e+09]])
'''
```

### 2. 底层揭秘：带下划线 `_` 的潜规则（面试必考）

这是 PyTorch 中极其重要的一个命名规范：**所有以 `_` 结尾的方法，都是 In-place（原地）操作。**

- **普通操作（如 `add`, `view`）**：会在显存中另外开辟一块新空间，或者返回一个新视角的张量，**不会修改原来的数据**。
    
- **In-place 操作（如 `add_`, `masked_fill_`）**：**极其暴力！它直接冲进原张量所在的物理内存地址，把原来的旧数字直接抹掉，写上新数字。**
    

**为什么要这么设计？为了极致榨干显存！** Transformer 的 Attention Score 矩阵通常极其庞大（比如 `[32, 8, 1024, 1024]`）。如果为了做一次掩码遮盖，还要在显存里复制一份这么大的矩阵，显存瞬间就会爆炸。 调用 `masked_fill_`，就是在原有的物理内存上直接打补丁，**零额外内存开销**！

### 3. 物理直觉：“漏印纸板 (Stencil)” 理论

你可以把 `masked_fill_` 想象成街头涂鸦艺术家使用的**漏印纸板**。

1. **你的原张量 (`scores`)**：是一面刚刚画好草图的白墙。
    
2. **你的掩码 (`mask`)**：是一张纸板，上面的 `True` 就是纸板上被镂空（剪掉）的洞，`False` 是实心的挡板。
    
3. **指定的数值 (`-1e9`)**：是你手里拿着的黑色喷漆。
    

你把纸板（`mask`）盖在墙上（`scores`），拿起喷漆（`-1e9`）对着墙面一阵狂喷。 拿走纸板后你会发现：只有纸板上有洞（True）的地方，才被染成了黑色（-1e9）；实心（False）的地方，完好地保留了原来的草图打分。

### 4. Transformer 中的两大经典战役

在实际的 NLP 任务中，这块“漏印纸板”通常有两种极其经典的刻法：

#### 战役 A：Padding Mask（除杂草）

- **场景**：因为在一个 Batch 里，句子的长度必须一致，不够长的句子会被强行补上 `<PAD>` 占位符。
    
- **动作**：我们在 `mask` 里把所有 `<PAD>` 对应的位置设为 `True`，用 `masked_fill_` 喷上 `-1e9`。
    
- **意义**：阻止 Attention 去关注毫无意义的填充符，让模型的算力全部集中在真实的单词上。
    

#### 战役 B：Causal Mask / Look-ahead Mask（防作弊）

- **场景**：在 Decoder 生成文字时，模型只能根据前面的词预测下一个词，绝对不能“偷看”未来的词。
    
- **动作**：刻一张下三角实心、上三角镂空（True）的纸板。
    
- **意义**：把右上角所有代表“未来信息”的打分全部变成 `-1e9`。到了 Softmax 这一步，这些未来词的关注概率就会变成绝对的 `0`，完美杜绝了信息泄露（穿越）。

---
## 三种掩码实现

在 transformer 中一共有三种掩码

- **第一种：Encoder 中的 padding mask**
	
	同一 batch 内不同样本长度不同，会使用 padding 补齐，但是这些 padding 的部分不应该被模型关注到，所以需要使用 padding mask 来屏蔽掉这些位置的影响
	
- 第二种：Decoder Casual Mask
	
	transformer 模型在训练时是并行输入的，但是在预测时，预测位置i只能关注到位置i之前的输入，所以需要使用 lookahead mask 来屏蔽掉位置i之后的输入
	
- **第三种：Encoder-Decoder Cross-Attention Mask**
	
	Decoder 的每层 Cross-Attention 中，需要拿自己的 Query 去和 Encoder 的 Key 和 Value
	
	做 Attention，但是 Encoder 的输入可能也有 padding，所以需要使用 Cross-Attention Mask 来屏蔽掉 Encoder 中的 padding 部分
### 1. padding 掩码

```python
## Encoder 中的 padding mask

def get_len_mask(batch_size: int, max_len: int, feat_lens: torch.Tensor, device: torch.device) -> torch.Tensor:

"""

获取长度掩码

  

输入：

batch_size: int, batch 大小

max_len: int, 序列最大长度

feat_lens: torch.Tensor, 每个样本的实际长度，形状为 (b,)

device: torch.device, 设备

输出：

mask: torch.Tensor, 长度掩码，形状为 (b, max_len, max_len)

  

注意力矩阵

原始输入形状 [batch_size, seq_len, embedding_dim]

但一般会对原始输入进行 padding 补齐到 max_len，所以输入形状变为 [batch_size, max_len, embedding_dim]

在计算注意力时，Query、Key、Value 的形状都是 [batch_size, max_len, embedding_dim]

注意力矩阵的形状为 [batch_size, max_len, max_len]

注意力得分矩阵的形状为 [batch_size, max_len, max_len]

掩码矩阵就是作用在注意力得分矩阵上的，掩码矩阵的形状也应该是 [batch_size, max_len, max_len]


"""

attn_mask = torch.ones((batch_size, max_len, max_len), device=device)

for i in range(batch_size):

attn_mask[i, :, :feat_lens[i]] = 0

return attn_mask.to(torch.bool)
```

### 2. 因果掩码

```python
## 因果掩码

def get_subsequent_mask(batch_size: int, max_len: int, device: torch.device) -> torch.Tensor:

"""

获取因果掩码

输入：

batch_size: int, batch 大小

max_len: int, 序列最大长度

device: torch.device, 设备

输出：

mask: torch.Tensor, 因果掩码，形状为 (b, max_len, max_len)

  

一个解码器层包含两种注意力机制：自注意力和交叉注意力。

自注意力机制允许解码器在生成每个位置的输出时关注输入序列中的所有位置，

而交叉注意力机制允许解码器在生成每个位置的输出时关注编码器输出序列中的所有位置。

  

因果掩码只作用在解码器的自注意力机制中，目的是为了确保解码器在生成每个位置的输出时只能关注输入序列中该位置之前的位置，

从而保证生成的输出序列是自回归的。

  

decoder 在训练时，输出序列长度和输出序列长度是一样的

  

decoder 输入 size 为 [batch_size, max_len, embedding_dim]

在 decoder 的 自注意力子层中

Query、Key、Value 的形状都是 [batch_size, max_len, embedding_dim]

注意力矩阵的形状为 [batch_size, max_len, max_len]

注意力得分矩阵的形状为 [batch_size, max_len, max_len]

故 因果掩码矩阵形状 为 [batch_size, max_len, max_len]

"""

  

return torch.triu(torch.ones((batch_size, max_len, max_len), device=device), diagonal=1).to(torch.bool)
```

### 3. 交叉注意力掩码

```python
## Encoder-Decoder Cross-Attention Mask

def get_enc_dec_mask(

batch_size: int,

max_feat_len: int,

feat_lens: torch.Tensor,

max_label_len: int,

device: torch.device

) -> torch.Tensor:

"""

获取 Encoder-Decoder Cross-Attention Mask

输入：

batch_size: int, batch 大小

max_feat_len: int, 编码器输入序列最大长度

feat_lens: torch.Tensor, 每个样本的实际长度，形状为 (b,)

max_label_len: int, 解码器输入序列最大长度

device: torch.device, 设备

输出：

mask: torch.Tensor, Encoder-Decoder Cross-Attention Mask，形状为 (b, max_label_len, max_feat_len)

  

decoder 的交叉注意力子层

输入形状为 [batch_size, max_label_len, embedding_dim]

Query 是 decoder 的输入计算出来的

Query 的形状为 [batch_size, max_label_len, embedding_dim]

  

拿 decoder 的 Query 去和 encoder 的 Key 和 Value 做 Attention

  

encoder 的输入形状为 [batch_size, max_feat_len, embedding_dim]

Key 和 Value 的形状为 [batch_size, max_feat_len, embedding_dim]

QxK^T 的形状为 [batch_size, max_label_len, max_feat_len]

"""

  

attn_mask = torch.zeros((batch_size, max_label_len, max_feat_len), device=device)

for i in range(batch_size):

attn_mask[i, :, feat_lens[i]:] = 1

return attn_mask.to(torch.bool)
```

---
## 多头注意力实现





