"""
中文逐行注释学习版。

原始文件：
beginner_source/introyt/introyt1_tutorial.py

说明：
这个文件保留了原教程的代码顺序，并用中文解释每个关键代码行的作用。
它用于学习阅读，不用于替代 PyTorch Tutorials 的正式 Sphinx Gallery 源文件。
原教程本身包含大量英文叙述和 Sphinx 标记；这里把重点放在可执行代码和核心概念上。
"""

# 导入 PyTorch 主包；后面所有张量创建、数学运算和模型计算都依赖 torch。
import torch

# 创建一个形状为 5 行 3 列的张量，里面所有元素都是 0。
z = torch.zeros(5, 3)
# 打印这个全 0 张量，方便观察它的形状和内容。
print(z)
# 打印张量的数据类型；默认情况下，torch.zeros 创建的是 float32 浮点数。
print(z.dtype)

# 创建一个形状为 5 行 3 列的全 1 张量，并显式指定数据类型为 16 位整数。
i = torch.ones((5, 3), dtype=torch.int16)
# 打印整数张量；输出里会显示 dtype=torch.int16，说明类型不是默认 float32。
print(i)

# 固定随机数种子，让后续随机生成的张量可以复现。
torch.manual_seed(1729)
# 生成一个 2 行 2 列的随机张量，元素默认在 [0, 1) 区间内。
r1 = torch.rand(2, 2)
# 打印提示文字，说明下面输出的是第一个随机张量。
print('A random tensor:')
# 打印第一个随机张量。
print(r1)

# 再生成一个 2 行 2 列的随机张量；因为随机数生成器已经往前走了，所以值会和 r1 不同。
r2 = torch.rand(2, 2)
# 打印提示文字；开头的 \n 用来多输出一个空行，让结果更清楚。
print('\nA different random tensor:')
# 打印第二个随机张量；它会产生新的随机值。
print(r2)

# 再次把随机数种子重置为 1729，相当于把随机数生成器调回同一个起点。
torch.manual_seed(1729)
# 重新生成一个 2 行 2 列的随机张量；因为种子相同，它应该和 r1 完全一样。
r3 = torch.rand(2, 2)
# 打印提示文字，说明下面这个张量应该和 r1 匹配。
print('\nShould match r1:')
# 打印第三个随机张量，用来验证固定随机种子的复现效果。
print(r3)

# 创建一个 2 行 3 列的全 1 张量。
ones = torch.ones(2, 3)
# 打印全 1 张量。
print(ones)

# 创建另一个 2 行 3 列的张量，再把每个元素乘以 2，因此得到全 2 张量。
twos = torch.ones(2, 3) * 2
# 打印全 2 张量。
print(twos)

# 对两个形状相同的张量做加法；PyTorch 会逐元素相加。
threes = ones + twos
# 打印相加结果；每个位置都是 1 + 2，所以结果全是 3。
print(threes)
# 打印结果张量的形状；它仍然是 2 行 3 列。
print(threes.shape)

# 生成一个形状为 2 行 3 列的随机张量。
r1 = torch.rand(2, 3)
# 生成一个形状为 3 行 2 列的随机张量。
r2 = torch.rand(3, 2)
# 如果取消下面这一行的注释，程序会报错，因为 2x3 和 3x2 不能直接逐元素相加。
# r3 = r1 + r2

# 先生成 [0, 1) 的 2x2 随机数，减去 0.5 后变成 [-0.5, 0.5)，再乘 2 得到约 [-1, 1)。
r = (torch.rand(2, 2) - 0.5) * 2
# 打印提示文字，说明下面是随机矩阵 r。
print('A random matrix, r:')
# 打印随机矩阵 r。
print(r)

# 打印提示文字，说明下面计算绝对值。
print('\nAbsolute value of r:')
# 对 r 中每个元素取绝对值；负数变正数，正数保持不变。
print(torch.abs(r))

# 打印提示文字，说明下面计算反正弦。
print('\nInverse sine of r:')
# 对 r 中每个元素计算反正弦；输入值需要落在 [-1, 1]，上面构造 r 正好满足这个范围。
print(torch.asin(r))

# 打印提示文字，说明下面计算矩阵行列式。
print('\nDeterminant of r:')
# 计算 2x2 矩阵 r 的行列式；行列式常用于线性代数中的可逆性、面积缩放等概念。
print(torch.det(r))
# 打印提示文字，说明下面计算奇异值分解。
print('\nSingular value decomposition of r:')
# 对矩阵做奇异值分解；torch.svd 是旧接口，新代码通常更推荐 torch.linalg.svd。
print(torch.svd(r))

# 打印提示文字，说明下面同时计算标准差和平均值。
print('\nAverage and standard deviation of r:')
# 计算 r 中所有元素的标准差和平均值；返回顺序是标准差、平均值。
print(torch.std_mean(r))
# 打印提示文字，说明下面计算最大值。
print('\nMaximum value of r:')
# 找出 r 中所有元素里的最大值。
print(torch.max(r))

# 再次导入 torch；原教程为了每个小节可以独立阅读，会重复 import。
import torch
# 导入神经网络模块；nn.Module、Conv2d、Linear、Loss 等都在这里。
import torch.nn as nn
# 导入常用的函数式接口；这里主要用 ReLU 和 max_pool2d。
import torch.nn.functional as F


# 定义一个 LeNet 风格的卷积神经网络；所有 PyTorch 模型通常都继承 nn.Module。
class LeNet(nn.Module):

    # 初始化函数负责创建网络层；这些层会保存可学习参数。
    def __init__(self):
        # 调用父类 nn.Module 的初始化逻辑，让 PyTorch 能正确跟踪子模块和参数。
        super(LeNet, self).__init__()
        # 第一层卷积：输入通道 1 个，输出通道 6 个，卷积核大小 5x5。
        self.conv1 = nn.Conv2d(1, 6, 5)
        # 第二层卷积：输入通道来自上一层的 6 个输出通道，输出通道 16 个，卷积核仍是 5x5。
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 第一层全连接：输入特征数是 16*5*5，输出特征数是 120。
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # 第二层全连接：把 120 维特征变成 84 维特征。
        self.fc2 = nn.Linear(120, 84)
        # 第三层全连接：把 84 维特征变成 10 维输出，对应 10 个类别的分数。
        self.fc3 = nn.Linear(84, 10)

    # forward 定义输入张量如何经过各层得到输出；调用 net(input) 时会自动进入这里。
    def forward(self, x):
        # 输入形状通常是 (batch, 1, 32, 32)；conv1 后变成 (batch, 6, 28, 28)，池化后变成 (batch, 6, 14, 14)。
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # conv2 后变成 (batch, 16, 10, 10)，再经过 2x2 池化后变成 (batch, 16, 5, 5)。
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # 把每张图的 16*5*5 个特征拉平成一行；-1 让 PyTorch 自动推断 batch 大小。
        x = x.view(-1, self.num_flat_features(x))
        # 经过第一层全连接并使用 ReLU 激活，增加非线性表达能力。
        x = F.relu(self.fc1(x))
        # 经过第二层全连接并使用 ReLU 激活。
        x = F.relu(self.fc2(x))
        # 输出 10 个原始分数；这里不手动 softmax，因为分类损失函数通常会内部处理。
        x = self.fc3(x)
        # 返回模型的最终输出。
        return x

    # 计算除 batch 维度以外的所有特征数量，用于把卷积特征图拉平成向量。
    def num_flat_features(self, x):
        # x.size() 返回所有维度；[1:] 去掉第 0 维 batch，只保留通道、高、宽等特征维度。
        size = x.size()[1:]
        # 初始化特征数为 1，后面逐个维度相乘。
        num_features = 1
        # 遍历每一个特征维度，例如 16、5、5。
        for s in size:
            # 累乘维度大小，例如 1*16*5*5 得到 400。
            num_features *= s
        # 返回单个样本被拉平后的特征总数。
        return num_features


# 创建 LeNet 模型实例；此时各层参数已经随机初始化，但还没有训练。
net = LeNet()
# 打印模型结构；PyTorch 会展示各个子层及其输入输出配置。
print(net)

# 创建一个模拟输入：1 张图片、1 个颜色通道、大小 32x32。
input = torch.rand(1, 1, 32, 32)
# 打印提示文字，说明下面展示输入 batch 的形状。
print('\nImage batch shape:')
# 打印输入形状；结果应为 torch.Size([1, 1, 32, 32])。
print(input.shape)

# 像调用函数一样调用模型；PyTorch 会自动执行 net.forward(input)。
output = net(input)
# 打印提示文字，说明下面是模型原始输出。
print('\nRaw output:')
# 打印模型输出；因为模型尚未训练，这些分数没有实际分类意义。
print(output)
# 打印输出形状；对 1 张图片输出 10 个类别分数，所以形状是 (1, 10)。
print(output.shape)

# 再次导入 torch；原教程按小节组织，所以重复 import 是为了教学清晰。
import torch
# 导入 torchvision；它提供常见视觉数据集、图像变换和工具函数。
import torchvision
# 导入 transforms；用于把图片转换成模型需要的张量格式。
import torchvision.transforms as transforms

# 定义一个图像预处理流水线；Compose 会按列表顺序依次执行每个变换。
transform = transforms.Compose(
    # ToTensor 把 PIL 图片或 numpy 数组转成形状为 (C, H, W) 的 torch.Tensor。
    [transforms.ToTensor(),
     # Normalize 对 RGB 三个通道分别做标准化：新值 = (原值 - mean) / std。
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])

# 创建 CIFAR-10 训练集对象；root 指定数据目录，train=True 表示训练集。
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        # download=True 表示本地没有数据时自动下载，transform 指定读取图片后的预处理。
                                        download=True, transform=transform)

# 用 DataLoader 包装 Dataset；它负责按 batch 取数据、打乱顺序和并行加载。
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          # batch_size=4 表示每次取 4 张图；shuffle=True 表示每轮训练前打乱顺序。
                                          shuffle=True, num_workers=0)

# 导入 matplotlib 的绘图接口，用于显示图片。
import matplotlib.pyplot as plt
# 导入 numpy；显示图片时要把张量转成 numpy 数组并调整维度顺序。
import numpy as np

# CIFAR-10 的 10 个类别名称；标签 0 到 9 会对应这里的字符串。
classes = ('plane', 'car', 'bird', 'cat',
           # 继续列出剩余类别；这个元组的顺序必须和 CIFAR-10 标签定义一致。
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 定义显示图片的辅助函数；输入 img 通常是一个形状为 (C, H, W) 的张量。
def imshow(img):
    # 反归一化，方便显示；注意这里严格对应 Normalize((0.5,), (0.5,))，对上面的 CIFAR 均值方差只是近似显示。
    img = img / 2 + 0.5
    # matplotlib 不能直接显示 PyTorch 张量，所以先转成 numpy 数组。
    npimg = img.numpy()
    # PyTorch 图片维度是 (C, H, W)，matplotlib 需要 (H, W, C)，因此用 transpose 调换维度。
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


# 把 DataLoader 转成迭代器；后面可以用 next 取出一个 batch。
dataiter = iter(trainloader)
# 从训练数据里取出一个 batch；images 是图片张量，labels 是对应类别编号。
images, labels = next(dataiter)

# make_grid 会把 4 张图片拼成一张网格图，再交给 imshow 显示。
imshow(torchvision.utils.make_grid(images))
# 逐个读取 batch 中 4 个标签，并把数字标签转换成类别名称打印出来。
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

# 再次导入 torch；下面进入完整训练小节。
import torch
# 导入神经网络模块。
import torch.nn as nn
# 导入函数式接口，例如 ReLU。
import torch.nn.functional as F
# 导入优化器模块；SGD、Adam 等优化算法都在这里。
import torch.optim as optim

# 再次导入 torchvision。
import torchvision
# 再次导入图像变换模块。
import torchvision.transforms as transforms

# 导入 matplotlib 主包；原教程中虽然导入了它，但主要使用 pyplot。
import matplotlib
# 导入 pyplot 用于画图。
import matplotlib.pyplot as plt
# 导入 numpy 用于图像张量到数组的转换。
import numpy as np

# 定义训练和测试都要使用的图像预处理流水线。
transform = transforms.Compose(
    # ToTensor 先把图片像素从 0-255 转成 0-1 浮点张量。
    [transforms.ToTensor(),
     # 这里用均值 0.5、标准差 0.5，把像素大致从 [0, 1] 映射到 [-1, 1]。
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# 创建 CIFAR-10 训练集；train=True 表示读取 50000 张训练图片。
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        # 本地没有数据时下载，并对每张图片应用 transform。
                                        download=True, transform=transform)
# 创建训练 DataLoader；每次给模型 4 张图，并在每个 epoch 打乱顺序。
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          # num_workers=2 表示使用 2 个子进程加载数据。
                                          shuffle=True, num_workers=0)

# 创建 CIFAR-10 测试集；train=False 表示读取 10000 张测试图片。
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       # 测试集也要使用和训练集一致的预处理，否则数据分布会不一致。
                                       download=True, transform=transform)
# 创建测试 DataLoader；测试时通常不需要打乱顺序，所以 shuffle=False。
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         # 测试集同样按每批 4 张图读取。
                                         shuffle=False, num_workers=0)

# 再次定义 CIFAR-10 的类别名称，后面打印标签时使用。
classes = ('plane', 'car', 'bird', 'cat',
           # 继续列出后 6 个类别。
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 再次导入 pyplot；原教程小节可独立运行，所以会重复导入。
import matplotlib.pyplot as plt
# 再次导入 numpy。
import numpy as np


# 定义显示图片的函数；这里的反归一化和上面的 Normalize((0.5,), (0.5,)) 是匹配的。
def imshow(img):
    # 把 [-1, 1] 范围的数据恢复到 [0, 1]，这样颜色显示更正常。
    img = img / 2 + 0.5
    # 把 torch.Tensor 转成 numpy.ndarray，供 matplotlib 使用。
    npimg = img.numpy()
    # 把通道维从最前面移动到最后面，符合 matplotlib 的图片格式要求。
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


# 创建训练 DataLoader 的迭代器。
dataiter = iter(trainloader)
# 取出一个训练 batch；images 形状通常是 (4, 3, 32, 32)，labels 形状是 (4,)。
images, labels = next(dataiter)

# 把一个 batch 的图片拼成网格并显示。
imshow(torchvision.utils.make_grid(images))
# 打印这个 batch 中 4 张图片的类别名称。
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


# 定义用于 CIFAR-10 分类的卷积神经网络；这是 LeNet 的彩色图片版本。
class Net(nn.Module):
    # 初始化函数创建网络层。
    def __init__(self):
        # 初始化 nn.Module 父类，让参数注册和子模块管理正常工作。
        super(Net, self).__init__()
        # 第一层卷积：输入 3 个通道，因为 CIFAR-10 是 RGB 彩色图；输出 6 个通道。
        self.conv1 = nn.Conv2d(3, 6, 5)
        # 定义一个 2x2 最大池化层，步幅也是 2，用于把宽高减半。
        self.pool = nn.MaxPool2d(2, 2)
        # 第二层卷积：输入 6 个通道，输出 16 个通道，卷积核 5x5。
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 第一层全连接：卷积部分最终得到 16 个 5x5 特征图，因此输入是 16*5*5。
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # 第二层全连接：120 维变成 84 维。
        self.fc2 = nn.Linear(120, 84)
        # 输出层：84 维变成 10 维，对应 CIFAR-10 的 10 个类别。
        self.fc3 = nn.Linear(84, 10)

    # 定义前向传播逻辑；训练和测试时都会执行这个函数。
    def forward(self, x):
        # 输入 (batch, 3, 32, 32)，经过 conv1、ReLU、pool 后变成 (batch, 6, 14, 14)。
        x = self.pool(F.relu(self.conv1(x)))
        # 再经过 conv2、ReLU、pool 后变成 (batch, 16, 5, 5)。
        x = self.pool(F.relu(self.conv2(x)))
        # 把每张图的特征展平成长度为 400 的向量，供全连接层处理。
        x = x.view(-1, 16 * 5 * 5)
        # 第一层全连接加 ReLU 激活。
        x = F.relu(self.fc1(x))
        # 第二层全连接加 ReLU 激活。
        x = F.relu(self.fc2(x))
        # 最后一层输出 10 个类别分数；分数最大的一类通常作为预测类别。
        x = self.fc3(x)
        # 返回输出分数。
        return x


# 创建要训练的模型实例。
net = Net()

# 定义交叉熵损失函数；多分类任务中非常常用，输入是类别分数和真实类别编号。
criterion = nn.CrossEntropyLoss()
# 定义 SGD 优化器；它会更新 net.parameters() 中所有可学习参数。
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# 外层循环表示训练轮数；range(2) 表示完整遍历训练集 2 次。
for epoch in range(2):

    # running_loss 用来累计一段时间内的损失，方便每隔一段打印平均值。
    running_loss = 0.0
    # 内层循环遍历训练 DataLoader；i 是 batch 编号，data 是一个 batch 的图片和标签。
    for i, data in enumerate(trainloader, 0):
        # 从 data 中拆出输入图片和真实标签。
        inputs, labels = data

        # 清空上一轮反向传播留下的梯度；PyTorch 默认会累加梯度，所以每个 batch 前都要清零。
        optimizer.zero_grad()

        # 前向传播：把输入图片送进模型，得到每张图片的 10 个类别分数。
        outputs = net(inputs)
        # 用预测分数和真实标签计算损失；损失越小，说明预测越接近正确答案。
        loss = criterion(outputs, labels)
        # 反向传播：根据损失计算每个可学习参数的梯度。
        loss.backward()
        # 优化器根据梯度更新参数；这是模型真正“学习”的一步。
        optimizer.step()

        # 把当前 batch 的损失值加到累计损失中；loss.item() 把单元素张量转成 Python 数字。
        running_loss += loss.item()
        # 每处理 2000 个 mini-batch 打印一次平均损失；i 从 0 开始，所以 1999 表示第 2000 个。
        if i % 2000 == 1999:
            # 打印当前 epoch、batch 编号和最近 2000 个 batch 的平均损失。
            print('[%d, %5d] loss: %.3f' %
                  # epoch + 1 让显示从第 1 轮开始；i + 1 让 batch 数从 1 开始。
                  (epoch + 1, i + 1, running_loss / 2000))
            # 打印后把累计损失清零，开始统计下一段。
            running_loss = 0.0

# 训练循环结束后打印完成提示。
print('Finished Training')

# correct 用来累计预测正确的样本数量。
correct = 0
# total 用来累计测试样本总数。
total = 0
# 测试阶段不需要计算梯度；no_grad 可以节省内存并加快推理。
with torch.no_grad():
    # 遍历测试集的每个 batch。
    for data in testloader:
        # 拆出测试图片和对应真实标签。
        images, labels = data
        # 前向传播得到模型对测试图片的类别分数。
        outputs = net(images)
        # 在类别维度上找最大分数；predicted 是每张图片预测出的类别编号。
        _, predicted = torch.max(outputs.data, 1)
        # labels.size(0) 是当前 batch 的样本数，把它加到总数里。
        total += labels.size(0)
        # predicted == labels 会得到布尔张量；sum 统计预测正确的数量，item 转成 Python 数字。
        correct += (predicted == labels).sum().item()

# 计算并打印测试集准确率；CIFAR-10 测试集共有 10000 张图片。
print('Accuracy of the network on the 10000 test images: %d %%' % (
    # correct / total 是正确率，乘以 100 转成百分比。
    100 * correct / total))
