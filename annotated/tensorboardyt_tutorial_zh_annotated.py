"""
`Introduction <introyt1_tutorial.html>`_ ||
`Tensors <tensors_deeper_tutorial.html>`_ ||
`Autograd <autogradyt_tutorial.html>`_ ||
`Building Models <modelsyt_tutorial.html>`_ ||
**TensorBoard Support** ||
`Training Models <trainingyt.html>`_ ||
`Model Understanding <captumyt.html>`_

PyTorch TensorBoard Support
===========================

中文注释版说明：
本文件基于 ``beginner_source/introyt/tensorboardyt_tutorial.py`` 重新创建。
原教程的英文说明、代码顺序和运行逻辑保持不变；新增的中文注释用于帮助初学者
理解如何准备 Fashion-MNIST 数据、训练一个简单卷积网络，并把图片、损失曲线、
模型计算图和数据嵌入写入 TensorBoard。

Follow along with the video below or on `youtube <https://www.youtube.com/watch?v=6CEld3hZgqc>`__.

.. raw:: html

   <div style="margin-top:10px; margin-bottom:10px;">
     <iframe width="560" height="315" src="https://www.youtube.com/embed/6CEld3hZgqc" frameborder="0" allow="accelerometer; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
   </div>

Before You Start
----------------

To run this tutorial, you’ll need to install PyTorch, TorchVision,
Matplotlib, and TensorBoard.

With ``pip``:

.. code-block:: sh

    pip install torch torchvision matplotlib tensorboard

Once the dependencies are installed, restart this notebook in the Python
environment where you installed them.


Introduction
------------
 
In this notebook, we’ll be training a variant of LeNet-5 against the
Fashion-MNIST dataset. Fashion-MNIST is a set of image tiles depicting
various garments, with ten class labels indicating the type of garment
depicted. 

"""

# PyTorch model and training necessities
# 导入 PyTorch 主包，以及构建模型、损失函数和优化器时常用的模块。
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Image datasets and image manipulation
# torchvision 提供常见视觉数据集、图像工具和预处理变换。
import torchvision
from torchvision.transforms import v2

# Image display
# Matplotlib 和 NumPy 只用于在教程中显示图片，训练本身不依赖它们。
import matplotlib.pyplot as plt
import numpy as np

# PyTorch TensorBoard support
# SummaryWriter 是 PyTorch 写 TensorBoard 日志的主要入口。
# 通过它可以记录图片、标量曲线、模型图、embedding 等内容。
from torch.utils.tensorboard import SummaryWriter

######################################################################
# Showing Images in TensorBoard
# -----------------------------
# 
# Let’s start by adding sample images from our dataset to TensorBoard:
# 

# Gather datasets and prepare them for consumption
# transform 定义了每张图片进入模型前要经过的处理流程。
# Fashion-MNIST 原始图片是灰度图，像素范围通常是 0 到 255；
# 神经网络更适合使用浮点张量和相对稳定的数值范围。
transform = v2.Compose([
    # 把 PIL 图片或数组转换为 PyTorch 图像张量，形状通常为 [通道, 高, 宽]。
    v2.ToImage(),
    # 转成 float32；scale=True 会把像素缩放到 [0, 1]。
    v2.ToDtype(torch.float32, scale=True),
    # 用均值 0.5、标准差 0.5 做归一化，把 [0, 1] 大致映射到 [-1, 1]。
    # 这样输入分布更居中，训练时通常更稳定。
    v2.Normalize((0.5,), (0.5,))])

# Store separate training and validations splits in ./data
# 训练集用于更新模型参数；download=True 表示本地没有数据时自动下载。
training_set = torchvision.datasets.FashionMNIST('./data',
    download=True,
    train=True,
    transform=transform)
# 验证集不参与参数更新，只用于观察模型在未见过数据上的表现。
validation_set = torchvision.datasets.FashionMNIST('./data',
    download=True,
    train=False,
    transform=transform)

# DataLoader 负责把数据集切成一个个小批次，并按需打乱、并行读取。
# batch_size=4 让每次迭代返回 4 张图片和 4 个标签，方便先展示图片网格。
training_loader = torch.utils.data.DataLoader(training_set,
                                              batch_size=4,
                                              shuffle=True,
                                              num_workers=2)


# 验证集通常不需要打乱，因为它只用于评估平均损失，不影响模型学习。
validation_loader = torch.utils.data.DataLoader(validation_set,
                                                batch_size=4,
                                                shuffle=False,
                                                num_workers=2)

# Class labels
# Fashion-MNIST 的标签是 0 到 9 的整数，这里把整数映射成人类可读的类别名。
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

# Helper function for inline image display
def matplotlib_imshow(img, one_channel=False):
    # 如果图片只有一个灰度通道，就把通道维压成二维图片，方便 Matplotlib 显示。
    if one_channel:
        img = img.mean(dim=0)
    # 前面的 Normalize 把像素从 [0, 1] 变成了 [-1, 1]。
    # 显示图片时需要反归一化回 [0, 1]，否则颜色会不直观。
    img = img / 2 + 0.5     # unnormalize
    # Matplotlib 接收 NumPy 数组，所以这里从张量转为数组。
    npimg = img.numpy()
    if one_channel:
        # 灰度图使用 Greys 色图显示。
        plt.imshow(npimg, cmap="Greys")
    else:
        # PyTorch 图像通常是 [C, H, W]，Matplotlib 需要 [H, W, C]。
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

# Extract a batch of 4 images
# iter(...) 创建一个数据迭代器，next(...) 取出第一个小批次。
dataiter = iter(training_loader)
images, labels = next(dataiter)

# Create a grid from the images and show them
# make_grid 会把多张图片拼成一张大图，适合一次性写入 TensorBoard。
img_grid = torchvision.utils.make_grid(images)
matplotlib_imshow(img_grid, one_channel=True)


########################################################################
# Above, we used TorchVision and Matplotlib to create a visual grid of a
# minibatch of our input data. Below, we use the ``add_image()`` call on
# ``SummaryWriter`` to log the image for consumption by TensorBoard, and
# we also call ``flush()`` to make sure it’s written to disk right away.
# 

# Default log_dir argument is "runs" - but it's good to be specific
# torch.utils.tensorboard.SummaryWriter is imported above
# 指定日志目录可以让不同实验互不覆盖，也方便在 TensorBoard 中对比。
writer = SummaryWriter('runs/fashion_mnist_experiment_1')

# Write image data to TensorBoard log dir
# 第一个参数是这张图片在 TensorBoard 里的名称；第二个参数是要记录的图片张量。
writer.add_image('Four Fashion-MNIST Images', img_grid)
# flush() 会把缓冲区里的事件立即写入磁盘，避免刚写完就打开 TensorBoard 时看不到。
writer.flush()

# To view, start TensorBoard on the command line with:
#   tensorboard --logdir=runs
# ...and open a browser tab to http://localhost:6006/


##########################################################################
# If you start TensorBoard at the command line and open it in a new
# browser tab (usually at `localhost:6006 <localhost:6006>`__), you should
# see the image grid under the IMAGES tab.
# 
# Graphing Scalars to Visualize Training
# --------------------------------------
# 
# TensorBoard is useful for tracking the progress and efficacy of your
# training. Below, we’ll run a training loop, track some metrics, and save
# the data for TensorBoard’s consumption.
# 
# Let’s define a model to categorize our image tiles, and an optimizer and
# loss function for training:
# 

class Net(nn.Module):
    def __init__(self):
        # 初始化 nn.Module 的内部状态。自定义模型一般都要先调用它。
        super().__init__()
        # 第一层卷积：输入是 1 个灰度通道，输出 6 个特征图，卷积核大小为 5x5。
        # 对 28x28 图片做不带 padding 的 5x5 卷积后，空间尺寸会变成 24x24。
        self.conv1 = nn.Conv2d(1, 6, 5)
        # 2x2 最大池化会把高和宽都减半，用更小的特征图保留局部最强响应。
        self.pool = nn.MaxPool2d(2, 2)
        # 第二层卷积接收 6 个输入通道，输出 16 个更高层的特征图。
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 两次卷积和池化后，28x28 图片会变成 16 个 4x4 特征图。
        # 展平后共有 16 * 4 * 4 = 256 个特征。
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        # 后面的全连接层逐步把视觉特征转换成类别判断所需的表示。
        self.fc2 = nn.Linear(120, 84)
        # 最后一层输出 10 个分数，对应 Fashion-MNIST 的 10 个类别。
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # forward 描述一批输入如何通过模型。
        # 输入 x 的形状一般是 [batch_size, 1, 28, 28]。
        x = self.pool(F.relu(self.conv1(x)))
        # 第二组“卷积 -> ReLU -> 池化”继续提取更抽象的局部图案。
        x = self.pool(F.relu(self.conv2(x)))
        # 把卷积得到的 4 维张量展平成二维：[batch_size, 256]。
        # -1 表示让 PyTorch 根据实际 batch 大小自动推断这一维。
        x = x.view(-1, 16 * 4 * 4)
        # 全连接层之间使用 ReLU 引入非线性，让模型能表达更复杂的关系。
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # 这里直接返回原始类别分数 logits。
        # CrossEntropyLoss 内部会处理 softmax，所以模型末尾不需要再加 softmax。
        x = self.fc3(x)
        return x
    

# 创建模型、损失函数和优化器。
net = Net()
# CrossEntropyLoss 常用于多分类任务，输入是 logits，目标是类别编号。
criterion = nn.CrossEntropyLoss()
# SGD 会根据梯度更新模型参数；momentum 可以帮助优化过程沿稳定方向前进。
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


##########################################################################
# Now let’s train a single epoch, and evaluate the training vs. validation
# set losses every 1000 batches:
# 

print(len(validation_loader))
# 这里只训练 1 个 epoch，用于演示如何把训练指标写入 TensorBoard。
for epoch in range(1):  # loop over the dataset multiple times
    # running_loss 暂存最近一段 mini-batch 的训练损失总和。
    running_loss = 0.0

    # enumerate(..., 0) 会同时给出批次编号 i 和批次数据 data。
    for i, data in enumerate(training_loader, 0):
        # basic training loop
        # inputs 是图片张量，labels 是对应的类别编号。
        inputs, labels = data
        # 每个 batch 反向传播前都要清空旧梯度。
        # PyTorch 默认会累加梯度，不清空会把上一个 batch 的梯度也算进去。
        optimizer.zero_grad()
        # 前向传播：模型根据输入图片给出每个类别的分数。
        outputs = net(inputs)
        # 计算预测分数和真实标签之间的差距。
        loss = criterion(outputs, labels)
        # 反向传播：根据 loss 计算每个可学习参数的梯度。
        loss.backward()
        # 优化器根据梯度实际更新参数。
        optimizer.step()

        running_loss += loss.item()
        if i % 1000 == 999:    # Every 1000 mini-batches...
            print(f'Batch {i + 1}')
            # Check against the validation set
            # 验证损失用于观察模型是否也能在未参与训练的数据上表现良好。
            running_vloss = 0.0
            
            # In evaluation mode some model specific operations can be omitted eg. dropout layer
            # eval() 会把模型切换到评估模式。
            # 本模型没有 dropout/batchnorm，但养成训练和评估显式切换的习惯很重要。
            net.eval() # Switching to evaluation mode, eg. turning off regularisation
            # 验证阶段只看结果，不更新参数；no_grad() 可以减少内存和计算开销。
            with torch.no_grad():
                for j, vdata in enumerate(validation_loader, 0):
                    vinputs, vlabels = vdata
                    voutputs = net(vinputs)
                    vloss = criterion(voutputs, vlabels)
                    running_vloss += vloss.item()
            # 验证结束后切回训练模式，后续 batch 才会按训练行为运行。
            net.train() # Switching back to training mode, eg. turning on regularisation
            
            # 训练损失取最近 1000 个 mini-batch 的平均值。
            avg_loss = running_loss / 1000
            # 验证损失取整个验证集所有 mini-batch 的平均值。
            avg_vloss = running_vloss / len(validation_loader)
            
            # Log the running loss averaged per batch
            # add_scalars 可以把多个相关曲线放在同一张图里，便于比较训练和验证损失。
            writer.add_scalars('Training vs. Validation Loss',
                            { 'Training' : avg_loss, 'Validation' : avg_vloss },
                            # global_step 是横轴位置。这里用“已经处理过的 batch 数”表示进度。
                            epoch * len(training_loader) + i)

            # 清零，开始统计下一个 1000 mini-batch 的训练损失。
            running_loss = 0.0
print('Finished Training')

# 确保训练期间记录的标量数据已经写入日志文件。
writer.flush()


#########################################################################
# Switch to your open TensorBoard and have a look at the SCALARS tab.
# 
# Visualizing Your Model
# ----------------------
# 
# TensorBoard can also be used to examine the data flow within your model.
# To do this, call the ``add_graph()`` method with a model and sample
# input:
# 

# Again, grab a single mini-batch of images
# 取一批真实图片作为示例输入，用来追踪模型计算图。
dataiter = iter(training_loader)
images, labels = next(dataiter)

# add_graph() will trace the sample input through your model,
# and render it as a graph.
# add_graph 会让 TensorBoard 展示输入如何流过各个层。
# 这对检查模型结构、确认张量连接关系很有帮助。
writer.add_graph(net, images)
writer.flush()


#########################################################################
# When you switch over to TensorBoard, you should see a GRAPHS tab.
# Double-click the “NET” node to see the layers and data flow within your
# model.
# 
# Visualizing Your Dataset with Embeddings
# ----------------------------------------
# 
# The 28-by-28 image tiles we’re using can be modeled as 784-dimensional
# vectors (28 \* 28 = 784). It can be instructive to project this to a
# lower-dimensional representation. The ``add_embedding()`` method will
# project a set of data onto the three dimensions with highest variance,
# and display them as an interactive 3D chart. The ``add_embedding()``
# method does this automatically by projecting to the three dimensions
# with highest variance.
# 
# Below, we’ll take a sample of our data, and generate such an embedding:
# 

# Select a random subset of data and corresponding labels
def select_n_random(data, labels, n=100):
    # 数据和标签必须一一对应，否则随机抽样后图片和类别会错位。
    assert len(data) == len(labels)

    # randperm 生成一个随机排列，用它可以从数据集中随机抽取样本。
    perm = torch.randperm(len(data))
    # 默认抽取 100 个样本，数量太大会让 TensorBoard projector 变慢。
    return data[perm][:n], labels[perm][:n]

# Extract a random subset of data
# training_set.data 是原始图片数据，training_set.targets 是对应标签。
# 这里没有使用 transform 后的数据，因为 embedding 只需要把图片展平成特征向量。
images, labels = select_n_random(training_set.data, training_set.targets)

# get the class labels for each image
# TensorBoard projector 可以显示 metadata；把数字标签转成文字后更容易观察类别聚类。
class_labels = [classes[label] for label in labels]

# log embeddings
# 每张 28x28 图片展平成 784 维向量，作为 embedding 的输入特征。
features = images.view(-1, 28 * 28)
# add_embedding 会把高维特征投影到低维空间，在 TensorBoard 的 PROJECTOR 中显示。
# metadata 是每个点的类别名，label_img 是每个点旁边显示的小图片。
writer.add_embedding(features,
                    metadata=class_labels,
                    label_img=images.unsqueeze(1))
writer.flush()
# 完成所有写入后关闭 writer，释放文件句柄。
writer.close()


#######################################################################
# Now if you switch to TensorBoard and select the PROJECTOR tab, you
# should see a 3D representation of the projection. You can rotate and
# zoom the model. Examine it at large and small scales, and see whether
# you can spot patterns in the projected data and the clustering of
# labels.
# 
# For better visibility, it’s recommended to:
# 
# - Select “label” from the “Color by” drop-down on the left.
# - Toggle the Night Mode icon along the top to place the
#   light-colored images on a dark background.
# 
# Other Resources
# ---------------
# 
# For more information, have a look at:
# 
# - PyTorch documentation on `torch.utils.tensorboard.SummaryWriter <https://pytorch.org/docs/stable/tensorboard.html?highlight=summarywriter>`__
# - Tensorboard tutorial content in the `PyTorch.org Tutorials <https://pytorch.org/tutorials/>`__ 
# - For more information about TensorBoard, see the `TensorBoard
#   documentation <https://www.tensorflow.org/tensorboard>`__
