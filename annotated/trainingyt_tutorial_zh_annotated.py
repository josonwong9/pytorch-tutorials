"""
`Introduction <introyt1_tutorial.html>`_ ||
`Tensors <tensors_deeper_tutorial.html>`_ ||
`Autograd <autogradyt_tutorial.html>`_ ||
`Building Models <modelsyt_tutorial.html>`_ ||
`TensorBoard Support <tensorboardyt_tutorial.html>`_ ||
**Training Models** ||
`Model Understanding <captumyt.html>`_

Training with PyTorch
=====================

中文注释版说明：
本文件基于 ``beginner_source/introyt/trainingyt.py`` 重新创建。
原教程的英文说明、代码顺序和运行逻辑保持不变；新增的中文注释用于帮助初学者
理解一次完整 PyTorch 训练流程中的关键步骤：准备数据、构建模型、定义损失函数、
选择优化器、执行训练循环、进行验证，以及保存表现最好的模型参数。

Follow along with the video below or on `youtube <https://www.youtube.com/watch?v=jF43_wj_DCQ>`__.

.. raw:: html

   <div style="margin-top:10px; margin-bottom:10px;">
     <iframe width="560" height="315" src="https://www.youtube.com/embed/jF43_wj_DCQ" frameborder="0" allow="accelerometer; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
   </div>

Introduction
------------

In past videos, we’ve discussed and demonstrated:

- Building models with the neural network layers and functions of the torch.nn module
- The mechanics of automated gradient computation, which is central to
  gradient-based model training 
- Using TensorBoard to visualize training progress and other activities

In this video, we’ll be adding some new tools to your inventory:

- We’ll get familiar with the dataset and dataloader abstractions, and how
  they ease the process of feeding data to your model during a training loop 
- We’ll discuss specific loss functions and when to use them
- We’ll look at PyTorch optimizers, which implement algorithms to adjust
  model weights based on the outcome of a loss function

Finally, we’ll pull all of these together and see a full PyTorch
training loop in action.


Dataset and DataLoader
----------------------
 
The ``Dataset`` and ``DataLoader`` classes encapsulate the process of
pulling your data from storage and exposing it to your training loop in
batches.

The ``Dataset`` is responsible for accessing and processing single
instances of data.
 
The ``DataLoader`` pulls instances of data from the ``Dataset`` (either
automatically or with a sampler that you define), collects them in
batches, and returns them for consumption by your training loop. The
``DataLoader`` works with all kinds of datasets, regardless of the type
of data they contain.
 
For this tutorial, we’ll be using the Fashion-MNIST dataset provided by
TorchVision. We use ``torchvision.transforms.v2.Normalize()`` to
zero-center and normalize the distribution of the image tile content,
and download both training and validation data splits.

""" 

# 导入 PyTorch 主包。后面会用它创建张量、定义优化器、关闭梯度计算，以及保存模型参数。
import torch
# torchvision 提供计算机视觉常用工具。这里主要使用内置的 Fashion-MNIST 数据集和图片网格工具。
import torchvision
# transforms.v2 是 torchvision 的新版图像预处理接口，可以把多个处理步骤组合成流水线。
from torchvision.transforms import v2

# PyTorch TensorBoard support
# SummaryWriter 用来把训练指标写入 TensorBoard 日志文件，方便在浏览器里查看曲线。
from torch.utils.tensorboard import SummaryWriter
# datetime 用来生成当前时间戳，避免不同训练运行写入同一个日志目录或模型文件名。
from datetime import datetime


# transform 定义每张图片进入模型前的预处理流程。
# Fashion-MNIST 的原始图片是 28x28 灰度图；模型更适合接收 float32 张量和稳定的数值范围。
transform = v2.Compose([
    # 把 PIL 图片或数组转换成 PyTorch 图像张量，形状通常是 [通道数, 高, 宽]。
    v2.ToImage(),
    # 转成 float32；scale=True 会把像素值从 0-255 缩放到 0-1。
    v2.ToDtype(torch.float32, scale=True),
    # 归一化：先减去均值 0.5，再除以标准差 0.5。
    # 对 0-1 的像素来说，这会把数值大致变为 -1 到 1，让输入更居中。
    v2.Normalize((0.5,), (0.5,))
])

# Create datasets for training & validation, download if necessary
# training_set 是训练集，用来让模型学习参数；download=True 表示本地没有数据时自动下载。
training_set = torchvision.datasets.FashionMNIST('./data', train=True, transform=transform, download=True)
# validation_set 是验证集，用来评估模型在未参与训练的数据上的表现，不能用于更新参数。
validation_set = torchvision.datasets.FashionMNIST('./data', train=False, transform=transform, download=True)

# Create data loaders for our datasets; shuffle for training, not for validation
# DataLoader 会按 batch_size 把 Dataset 组织成一批批数据，训练循环每次读取一个小批次。
# 训练时 shuffle=True 可以打乱样本顺序，减少模型记住固定顺序带来的偏差。
training_loader = torch.utils.data.DataLoader(training_set, batch_size=4, shuffle=True)
# 验证时不需要打乱顺序，因为我们只关心整体平均损失，不会根据顺序更新模型。
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=4, shuffle=False)

# Class labels
# Fashion-MNIST 的标签是 0 到 9 的整数。这个元组把数字标签映射成更容易读懂的类别名。
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

# Report split sizes
# 打印训练集和验证集大小，确认数据已经正确加载，也方便理解每个 epoch 要处理多少样本。
print(f'Training set has {len(training_set)} instances')
print(f'Validation set has {len(validation_set)} instances')


######################################################################
# As always, let’s visualize the data as a sanity check:
# 

# Matplotlib 用于在教程中显示图片；NumPy 用于把张量转成 Matplotlib 更熟悉的数组格式。
import matplotlib.pyplot as plt
import numpy as np

# Helper function for inline image display
def matplotlib_imshow(img, one_channel=False):
    # Fashion-MNIST 是灰度图，只有一个通道。one_channel=True 时，把通道维合并成二维图像。
    if one_channel:
        img = img.mean(dim=0)
    # 前面 Normalize 把图片从 [0, 1] 转到了大约 [-1, 1]。
    # 显示图片前要反归一化回 [0, 1]，否则图像会过暗或过亮。
    img = img / 2 + 0.5     # unnormalize
    # Matplotlib 不能直接显示 PyTorch 张量，所以转换成 NumPy 数组。
    npimg = img.numpy()
    if one_channel:
        # 灰度图使用 Greys 色图显示。
        plt.imshow(npimg, cmap="Greys")
    else:
        # 彩色图片在 PyTorch 中通常是 [C, H, W]，而 Matplotlib 需要 [H, W, C]。
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

# 从训练 DataLoader 创建一个迭代器，方便手动取出第一批数据。
dataiter = iter(training_loader)
# next(dataiter) 返回一个 batch，其中 images 是图片张量，labels 是对应的类别编号。
images, labels = next(dataiter)

# Create a grid from the images and show them
# make_grid 会把多张图片拼成一张网格图，便于一次性查看一个小批次里的样本。
img_grid = torchvision.utils.make_grid(images)
matplotlib_imshow(img_grid, one_channel=True)
# labels[j] 是整数类别编号，classes[...] 会把它转换成可读的类别名称。
print('  '.join(classes[labels[j]] for j in range(4)))


#########################################################################
# The Model
# ---------
# 
# The model we’ll use in this example is a variant of LeNet-5 - it should
# be familiar if you’ve watched the previous videos in this series.
# 

# torch.nn 提供神经网络层、损失函数和 Module 基类。
import torch.nn as nn
# torch.nn.functional 提供无状态函数，例如 relu。它们不会保存可学习参数。
import torch.nn.functional as F

# PyTorch models inherit from torch.nn.Module
# 自定义模型一般继承 nn.Module，这样 PyTorch 才能自动追踪子层、参数和训练状态。
class GarmentClassifier(nn.Module):
    def __init__(self):
        # 初始化 nn.Module 内部机制。定义子层前必须调用，否则参数不会被正确注册。
        super().__init__()
        # 第一层卷积：输入通道为 1，因为 Fashion-MNIST 是灰度图；输出 6 个特征图。
        # kernel_size=5 表示使用 5x5 的卷积核扫描图片。
        self.conv1 = nn.Conv2d(1, 6, 5)
        # 最大池化层：窗口大小 2x2，步幅 2。它会把高和宽大约缩小一半。
        self.pool = nn.MaxPool2d(2, 2)
        # 第二层卷积：接收上一层的 6 个通道，输出 16 个更高级的特征图。
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 全连接层接收卷积部分展开后的特征。
        # 经过两次 5x5 卷积和两次 2x2 池化后，28x28 图片变成 16 个 4x4 特征图。
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        # 继续用全连接层把 120 个特征压缩成 84 个特征。
        self.fc2 = nn.Linear(120, 84)
        # 最后一层输出 10 个分数，对应 Fashion-MNIST 的 10 个类别。
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 第一段：卷积提取局部特征，ReLU 加入非线性，池化降低空间尺寸。
        x = self.pool(F.relu(self.conv1(x)))
        # 第二段重复同样结构，提取更高层次的图像特征。
        x = self.pool(F.relu(self.conv2(x)))
        # 全连接层需要二维输入 [batch_size, features]。
        # -1 让 PyTorch 自动推断 batch_size；16 * 4 * 4 是每张图片展开后的特征数。
        x = x.view(-1, 16 * 4 * 4)
        # 第一个全连接层后接 ReLU，帮助模型学习非线性关系。
        x = F.relu(self.fc1(x))
        # 第二个全连接层后继续接 ReLU。
        x = F.relu(self.fc2(x))
        # 输出层不接 softmax，因为 CrossEntropyLoss 会在内部处理类别分数。
        x = self.fc3(x)
        return x
    

# 创建模型实例。此时模型参数已经存在，但还没有经过训练。
model = GarmentClassifier()


##########################################################################
# Loss Function
# -------------
# 
# For this example, we’ll be using a cross-entropy loss. For demonstration
# purposes, we’ll create batches of dummy output and label values, run
# them through the loss function, and examine the result.
# 

# CrossEntropyLoss 常用于多分类任务。
# 它接收模型输出的原始类别分数 logits，以及真实类别编号，不需要手动做 softmax。
loss_fn = torch.nn.CrossEntropyLoss()

# NB: Loss functions expect data in batches, so we're creating batches of 4
# Represents the model's confidence in each of the 10 classes for a given input
# 这里构造 4 个样本的假输出，每个样本有 10 个类别分数。
# 数值越大，表示模型越倾向于把该样本判为对应类别。
dummy_outputs = torch.rand(4, 10)
# Represents the correct class among the 10 being tested
# 假标签同样有 4 个整数，每个整数表示对应样本的正确类别。
dummy_labels = torch.tensor([1, 5, 3, 7])
    
print(dummy_outputs)
print(dummy_labels)

# 损失函数会比较“模型给出的类别分数”和“真实类别”，返回这个 batch 的平均损失。
# 损失越小，说明模型输出越接近真实标签。
loss = loss_fn(dummy_outputs, dummy_labels)
print(f'Total loss for this batch: {loss.item()}')


#################################################################################
# Optimizer
# ---------
# 
# For this example, we’ll be using simple `stochastic gradient
# descent <https://pytorch.org/docs/stable/optim.html>`__ with momentum.
# 
# It can be instructive to try some variations on this optimization
# scheme:
# 
# - Learning rate determines the size of the steps the optimizer
#   takes. What does a different learning rate do to the your training
#   results, in terms of accuracy and convergence time?
# - Momentum nudges the optimizer in the direction of strongest gradient over
#   multiple steps. What does changing this value do to your results? 
# - Try some different optimization algorithms, such as averaged SGD, Adagrad, or
#   Adam. How do your results differ?
# 

# Optimizers specified in the torch.optim package
# 优化器负责根据反向传播得到的梯度更新模型参数。
# model.parameters() 会把模型里所有可学习参数交给优化器管理。
# lr 是学习率，控制每次更新的步子大小；momentum 会利用过去梯度方向，让更新更平滑。
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


#######################################################################################
# The Training Loop
# -----------------
# 
# Below, we have a function that performs one training epoch. It
# enumerates data from the DataLoader, and on each pass of the loop does
# the following:
# 
# - Gets a batch of training data from the DataLoader
# - Zeros the optimizer’s gradients 
# - Performs an inference - that is, gets predictions from the model for an input batch
# - Calculates the loss for that set of predictions vs. the labels on the dataset
# - Calculates the backward gradients over the learning weights
# - Tells the optimizer to perform one learning step - that is, adjust the model’s
#   learning weights based on the observed gradients for this batch, according to the
#   optimization algorithm we chose
# - It reports on the loss for every 1000 batches.
# - Finally, it reports the average per-batch loss for the last
#   1000 batches, for comparison with a validation run
# 

def train_one_epoch(epoch_index, tb_writer):
    # running_loss 用来累计最近一段 batch 的损失，便于计算平均值。
    running_loss = 0.
    # last_loss 保存最近一次汇报的平均损失，函数结束时返回它。
    last_loss = 0.
    
    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    # enumerate 会同时给出 batch 编号 i 和 batch 数据 data。
    # 这样可以每隔固定数量的 batch 打印一次训练进度。
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        # DataLoader 返回的 data 是一组输入图片和对应标签。
        # inputs 形状大致是 [batch_size, 1, 28, 28]，labels 形状是 [batch_size]。
        inputs, labels = data
        
        # Zero your gradients for every batch!
        # PyTorch 默认会累加梯度，所以每个 batch 开始前都要清空上一轮梯度。
        optimizer.zero_grad()
        
        # Make predictions for this batch
        # 前向传播：把图片输入模型，得到每张图片对 10 个类别的预测分数。
        outputs = model(inputs)
        
        # Compute the loss and its gradients
        # 计算预测分数和真实标签之间的损失。
        loss = loss_fn(outputs, labels)
        # 反向传播：根据损失计算每个可学习参数的梯度。
        loss.backward()
        
        # Adjust learning weights
        # 优化器读取参数梯度，并按照 SGD + momentum 的规则更新参数。
        optimizer.step()
        
        # Gather data and report
        # loss.item() 把只有一个值的张量转成 Python 数字，便于累加和打印。
        running_loss += loss.item()
        # 每 1000 个 batch 汇报一次平均训练损失。
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print(f'  batch {i + 1} loss: {last_loss}')
            # tb_x 是 TensorBoard 横轴上的步数。
            # 用 epoch_index * len(training_loader) 加上当前 batch 编号，可以得到全局训练步数。
            tb_x = epoch_index * len(training_loader) + i + 1
            # 把训练损失写入 TensorBoard，标签名是 Loss/train。
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            # 清零累计损失，开始统计下一个 1000 batch。
            running_loss = 0.
            
    return last_loss


##################################################################################
# Per-Epoch Activity
# ~~~~~~~~~~~~~~~~~~
# 
# There are a couple of things we’ll want to do once per epoch: 
#
# - Perform validation by checking our relative loss on a set of data that was not
#   used for training, and report this 
# - Save a copy of the model
# 
# Here, we’ll do our reporting in TensorBoard. This will require going to
# the command line to start TensorBoard, and opening it in another browser
# tab.
# 

# Initializing in a separate cell so we can easily add more epochs to the same run
# 用时间戳区分不同训练运行，避免 TensorBoard 日志目录互相覆盖。
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
# SummaryWriter 会把事件文件写到 runs/fashion_trainer_时间戳 目录下。
writer = SummaryWriter(f'runs/fashion_trainer_{timestamp}')
# epoch_number 记录当前是第几个 epoch，用于打印、日志横轴和模型文件名。
epoch_number = 0

# EPOCHS 表示完整遍历训练集的次数。这里训练 5 轮。
EPOCHS = 5

# 记录目前见过的最低验证损失。初始值设置得很大，保证第一轮通常会保存模型。
best_vloss = 1_000_000.

for epoch in range(EPOCHS):
    print(f'EPOCH {epoch_number + 1}:')
    
    # Make sure gradient tracking is on, and do a pass over the data
    # train(True) 会把模型切换到训练模式。
    # 对本模型影响不大，但对于 Dropout、BatchNorm 等层非常重要。
    model.train(True)
    # 执行一个 epoch 的训练，并拿到这一轮最后一次汇报的平均训练损失。
    avg_loss = train_one_epoch(epoch_number, writer)
    

    # running_vloss 用来累加验证集所有 batch 的损失。
    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population 
    # statistics for batch normalization.
    # eval() 把模型切换到评估模式。
    # 即使当前模型没有 Dropout 或 BatchNorm，也建议验证和推理时保持这个习惯。
    model.eval()

    # Disable gradient computation and reduce memory consumption.
    # 验证阶段只看模型表现，不更新参数，所以不需要保存梯度计算图。
    # torch.no_grad() 可以减少内存占用，并让验证速度更快。
    with torch.no_grad():
        for i, vdata in enumerate(validation_loader):
            # 验证数据同样是一批图片和对应标签。
            vinputs, vlabels = vdata
            # 前向传播得到验证集预测结果。
            voutputs = model(vinputs)
            # 计算该验证 batch 的损失。
            vloss = loss_fn(voutputs, vlabels)
            # 累加验证损失，稍后除以 batch 数得到平均值。
            running_vloss += vloss
    
    # i 从 0 开始计数，所以验证 batch 总数是 i + 1。
    avg_vloss = running_vloss / (i + 1)
    print(f'LOSS train {avg_loss} valid {avg_vloss}')
    
    # Log the running loss averaged per batch
    # for both training and validation
    # 同时记录训练损失和验证损失，方便在 TensorBoard 中比较两条曲线。
    # 如果训练损失下降但验证损失上升，可能说明模型开始过拟合。
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number + 1)
    # flush() 确保日志尽快写入磁盘，避免程序提前结束时丢失最近记录。
    writer.flush()
    
    # Track best performance, and save the model's state
    # 只在验证损失创新低时保存模型参数。
    # 这样得到的文件通常对应当前训练过程中泛化表现最好的模型。
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        # 文件名包含时间戳和 epoch 编号，便于区分不同训练运行和保存点。
        model_path = f'model_{timestamp}_{epoch_number}'
        # state_dict 只保存模型参数，不保存模型类定义。
        # 加载时仍然需要先创建同样结构的 GarmentClassifier。
        torch.save(model.state_dict(), model_path)
    
    # 进入下一个 epoch 前更新编号。
    epoch_number += 1


#########################################################################
# To load a saved version of the model:
#
# .. code:: python
#
#     saved_model = GarmentClassifier()
#     saved_model.load_state_dict(torch.load(PATH))
#
# Once you’ve loaded the model, it’s ready for whatever you need it for -
# more training, inference, or analysis.
# 
# Note that if your model has constructor parameters that affect model
# structure, you’ll need to provide them and configure the model
# identically to the state in which it was saved.
# 
# Other Resources
# ---------------
# 
# -  Docs on the `data
#    utilities <https://pytorch.org/docs/stable/data.html>`__, including
#    Dataset and DataLoader, at pytorch.org
# -  A `note on the use of pinned
#    memory <https://pytorch.org/docs/stable/notes/cuda.html#cuda-memory-pinning>`__
#    for GPU training
# -  Documentation on the datasets available in
#    `TorchVision <https://pytorch.org/vision/stable/datasets.html>`__,
#    `TorchText <https://pytorch.org/text/stable/datasets.html>`__, and
#    `TorchAudio <https://pytorch.org/audio/stable/datasets.html>`__
# -  Documentation on the `loss
#    functions <https://pytorch.org/docs/stable/nn.html#loss-functions>`__
#    available in PyTorch
# -  Documentation on the `torch.optim
#    package <https://pytorch.org/docs/stable/optim.html>`__, which
#    includes optimizers and related tools, such as learning rate
#    scheduling
# -  A detailed `tutorial on saving and loading
#    models <https://pytorch.org/tutorials/beginner/saving_loading_models.html>`__
# -  The `Tutorials section of
#    pytorch.org <https://pytorch.org/tutorials/>`__ contains tutorials on
#    a broad variety of training tasks, including classification in
#    different domains, generative adversarial networks, reinforcement
#    learning, and more 
# 
