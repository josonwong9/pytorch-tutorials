"""
`Introduction <introyt1_tutorial.html>`_ ||
`Tensors <tensors_deeper_tutorial.html>`_ ||
`Autograd <autogradyt_tutorial.html>`_ ||
**Building Models** ||
`TensorBoard Support <tensorboardyt_tutorial.html>`_ ||
`Training Models <trainingyt.html>`_ ||
`Model Understanding <captumyt.html>`_

Building Models with PyTorch
============================

中文注释版说明：
本文件基于 ``beginner_source/introyt/modelsyt_tutorial.py`` 重新创建。
原教程的英文说明和代码顺序被保留下来；新增的中文注释用于帮助初学者理解
每一段模型代码在做什么，以及张量形状、层参数和常见模块之间的关系。

Follow along with the video below or on `youtube <https://www.youtube.com/watch?v=OSqIP-mOWOI>`__.

.. raw:: html

   <div style="margin-top:10px; margin-bottom:10px;">
     <iframe width="560" height="315" src="https://www.youtube.com/embed/OSqIP-mOWOI" frameborder="0" allow="accelerometer; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
   </div>

``torch.nn.Module`` and ``torch.nn.Parameter``
----------------------------------------------

In this video, we’ll be discussing some of the tools PyTorch makes
available for building deep learning networks.

Except for ``Parameter``, the classes we discuss in this video are all
subclasses of ``torch.nn.Module``. This is the PyTorch base class meant
to encapsulate behaviors specific to PyTorch Models and their
components.

One important behavior of ``torch.nn.Module`` is registering parameters.
If a particular ``Module`` subclass has learning weights, these weights
are expressed as instances of ``torch.nn.Parameter``. The ``Parameter``
class is a subclass of ``torch.Tensor``, with the special behavior that
when they are assigned as attributes of a ``Module``, they are added to
the list of that modules parameters. These parameters may be accessed
through the ``parameters()`` method on the ``Module`` class.

As a simple example, here’s a very simple model with two linear layers
and an activation function. We’ll create an instance of it and ask it to
report on its parameters:

"""

# 导入 PyTorch 主包。这里主要会用到 torch.nn 中的神经网络层和模型基类。
import torch


# 所有自定义 PyTorch 模型通常都继承 torch.nn.Module。
# 继承 Module 后，PyTorch 才能自动追踪模型里的子层、参数和训练/推理状态。
class TinyModel(torch.nn.Module):

    def __init__(self):
        # super().__init__() 会初始化 Module 内部的登记机制。
        # 只有先调用它，后面赋给 self 的 Linear、ReLU 等子模块才会被正确注册。
        super().__init__()

        # 第一个全连接层：每个样本有 100 个输入特征，输出 200 个隐藏特征。
        # Linear 层内部包含 weight 和 bias，它们都是可学习参数。
        self.linear1 = torch.nn.Linear(100, 200)
        # ReLU 是非线性激活函数：小于 0 的值变成 0，大于 0 的值保持不变。
        # 如果模型只有 Linear 层，多层线性变换可以合并成一层，表达能力会很有限。
        self.activation = torch.nn.ReLU()
        # 第二个全连接层：把 200 个隐藏特征压到 10 个输出。
        # 在分类任务中，10 常常表示 10 个类别的得分。
        self.linear2 = torch.nn.Linear(200, 10)
        # Softmax 会把一组得分转换成概率分布。
        # dim=1 表示沿着“类别”这一维做归一化；常见输入形状是 [batch_size, num_classes]。
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        # forward() 定义数据如何流过模型。调用 tinymodel(x) 时，PyTorch 会自动执行它。
        # x 通常是一批样本，形状可以理解为 [批大小, 特征数]。
        x = self.linear1(x)
        # 加入非线性，让模型可以学习更复杂的函数关系。
        x = self.activation(x)
        # 继续把隐藏特征映射为类别得分。
        x = self.linear2(x)
        # 把类别得分转成每个类别的概率。
        x = self.softmax(x)
        return x


# 创建模型实例。此时 __init__() 中定义的层和参数已经完成注册。
tinymodel = TinyModel()

print('The model:')
# 打印整个模型，可以看到子模块的层级结构。
print(tinymodel)

print('\n\nJust one layer:')
# 也可以只打印某一个子层，检查这一层的输入/输出维度。
print(tinymodel.linear2)

print('\n\nModel params:')
# parameters() 会遍历模型中所有已注册的可学习参数。
# 对 TinyModel 来说，主要是 linear1 和 linear2 的 weight、bias。
for param in tinymodel.parameters():
    print(param)

print('\n\nLayer params:')
# 子层也有自己的 parameters()。这里只查看 linear2 的 weight 和 bias。
for param in tinymodel.linear2.parameters():
    print(param)


#########################################################################
# This shows the fundamental structure of a PyTorch model: there is an
# ``__init__()`` method that defines the layers and other components of a
# model, and a ``forward()`` method where the computation gets done. Note
# that we can print the model, or any of its submodules, to learn about
# its structure.
#
# Common Layer Types
# ------------------
#
# Linear Layers
# ~~~~~~~~~~~~~
#
# The most basic type of neural network layer is a *linear* or *fully
# connected* layer. This is a layer where every input influences every
# output of the layer to a degree specified by the layer’s weights. If a
# model has *m* inputs and *n* outputs, the weights will be an *m* x *n*
# matrix. For example:
#

# 创建一个线性层：输入特征数为 3，输出特征数为 2。
# 它会学习一个形状为 [2, 3] 的 weight 和一个形状为 [2] 的 bias。
lin = torch.nn.Linear(3, 2)
# 构造 1 个样本，每个样本有 3 个特征；形状是 [1, 3]。
x = torch.rand(1, 3)
print('Input:')
print(x)

print('\n\nWeight and Bias parameters:')
# 线性层的参数默认 requires_grad=True，训练时会通过反向传播更新。
for param in lin.parameters():
    print(param)

# 对输入做一次前向计算。数学形式可以理解为 y = x @ weight.T + bias。
y = lin(x)
print('\n\nOutput:')
print(y)


#########################################################################
# If you do the matrix multiplication of ``x`` by the linear layer’s
# weights, and add the biases, you’ll find that you get the output vector
# ``y``.
#
# One other important feature to note: When we checked the weights of our
# layer with ``lin.weight``, it reported itself as a ``Parameter`` (which
# is a subclass of ``Tensor``), and let us know that it’s tracking
# gradients with autograd. This is a default behavior for ``Parameter``
# that differs from ``Tensor``.
#
# Linear layers are used widely in deep learning models. One of the most
# common places you’ll see them is in classifier models, which will
# usually have one or more linear layers at the end, where the last layer
# will have *n* outputs, where *n* is the number of classes the classifier
# addresses.
#
# Convolutional Layers
# ~~~~~~~~~~~~~~~~~~~~
#
# *Convolutional* layers are built to handle data with a high degree of
# spatial correlation. They are very commonly used in computer vision,
# where they detect close groupings of features which the compose into
# higher-level features. They pop up in other contexts too - for example,
# in NLP applications, where a word’s immediate context (that is, the
# other words nearby in the sequence) can affect the meaning of a
# sentence.
#
# We saw convolutional layers in action in LeNet5 in an earlier video:
#

# functional 模块提供函数式接口，例如 F.relu()、F.max_pool2d()。
# 它们不保存自己的参数，适合用来表示简单的计算步骤。
import torch.nn.functional as F


# LeNet 是经典卷积神经网络。这里用它展示卷积、池化和全连接层如何组合。
class LeNet(torch.nn.Module):

    def __init__(self):
        super().__init__()
        # Conv2d 的三个核心参数是：输入通道数、输出通道数、卷积核大小。
        # 这里输入是黑白图像，所以只有 1 个通道；输出 6 个通道，表示学习 6 组特征。
        # kernel_size=5 表示每个卷积核观察 5x5 的局部区域。
        self.conv1 = torch.nn.Conv2d(1, 6, 5)
        # 第二个卷积层接收上一层的 6 个特征通道，再提取 16 个更高层特征。
        # kernel_size=3 表示使用 3x3 的窗口扫描输入。
        self.conv2 = torch.nn.Conv2d(6, 16, 3)
        # 全连接层又叫仿射变换，数学形式是 y = Wx + b。
        # 经过两次卷积和池化后，每个样本会变成 16 个 6x6 特征图。
        # 因此展平后的输入特征数是 16 * 6 * 6 = 576。
        self.fc1 = torch.nn.Linear(16 * 6 * 6, 120)
        # 继续压缩特征维度，为最终分类做准备。
        self.fc2 = torch.nn.Linear(120, 84)
        # 最后一层输出 10 个值，通常表示 10 个类别的原始得分 logits。
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        # 先卷积，再 ReLU，再最大池化。
        # max_pool2d(..., (2, 2)) 会在每个 2x2 区域中保留最大值，从而降低空间尺寸。
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # 如果池化窗口是正方形，可以只写一个数字 2，含义等同于 (2, 2)。
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # 卷积输出是多维特征图，全连接层需要二维输入 [batch_size, features]。
        # view() 用来改变张量形状，-1 表示让 PyTorch 自动推断这一维的大小。
        x = x.view(-1, self.num_flat_features(x))
        # 全连接层之间继续使用 ReLU 增加非线性表达能力。
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # 这里不做 softmax，很多损失函数（如 CrossEntropyLoss）会直接接收 logits。
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        # x.size() 返回张量每个维度的大小。
        # 第 0 维通常是 batch 维，展平特征时只需要计算后面各维的乘积。
        size = x.size()[1:]
        num_features = 1
        # 例如形状 [batch, 16, 6, 6] 会得到 16 * 6 * 6 = 576。
        for s in size:
            num_features *= s
        return num_features


##########################################################################
# Let’s break down what’s happening in the convolutional layers of this
# model. Starting with ``conv1``:
#
# -  LeNet5 is meant to take in a 1x32x32 black & white image. **The first
#    argument to a convolutional layer’s constructor is the number of
#    input channels.** Here, it is 1. If we were building this model to
#    look at 3-color channels, it would be 3.
# -  A convolutional layer is like a window that scans over the image,
#    looking for a pattern it recognizes. These patterns are called
#    *features,* and one of the parameters of a convolutional layer is the
#    number of features we would like it to learn. **This is the second
#    argument to the constructor is the number of output features.** Here,
#    we’re asking our layer to learn 6 features.
# -  Just above, I likened the convolutional layer to a window - but how
#    big is the window? **The third argument is the window or kernel
#    size.** Here, the “5” means we’ve chosen a 5x5 kernel. (If you want a
#    kernel with height different from width, you can specify a tuple for
#    this argument - e.g., ``(3, 5)`` to get a 3x5 convolution kernel.)
#
# The output of a convolutional layer is an *activation map* - a spatial
# representation of the presence of features in the input tensor.
# ``conv1`` will give us an output tensor of 6x28x28; 6 is the number of
# features, and 28 is the height and width of our map. (The 28 comes from
# the fact that when scanning a 5-pixel window over a 32-pixel row, there
# are only 28 valid positions.)
#
# We then pass the output of the convolution through a ReLU activation
# function (more on activation functions later), then through a max
# pooling layer. The max pooling layer takes features near each other in
# the activation map and groups them together. It does this by reducing
# the tensor, merging every 2x2 group of cells in the output into a single
# cell, and assigning that cell the maximum value of the 4 cells that went
# into it. This gives us a lower-resolution version of the activation map,
# with dimensions 6x14x14.
#
# Our next convolutional layer, ``conv2``, expects 6 input channels
# (corresponding to the 6 features sought by the first layer), has 16
# output channels, and a 3x3 kernel. It puts out a 16x12x12 activation
# map, which is again reduced by a max pooling layer to 16x6x6. Prior to
# passing this output to the linear layers, it is reshaped to a 16 \* 6 \*
# 6 = 576-element vector for consumption by the next layer.
#
# There are convolutional layers for addressing 1D, 2D, and 3D tensors.
# There are also many more optional arguments for a conv layer
# constructor, including stride length(e.g., only scanning every second or
# every third position) in the input, padding (so you can scan out to the
# edges of the input), and more. See the
# `documentation <https://pytorch.org/docs/stable/nn.html#convolution-layers>`__
# for more information.
#
# Recurrent Layers
# ~~~~~~~~~~~~~~~~
#
# *Recurrent neural networks* (or *RNNs)* are used for sequential data -
# anything from time-series measurements from a scientific instrument to
# natural language sentences to DNA nucleotides. An RNN does this by
# maintaining a *hidden state* that acts as a sort of memory for what it
# has seen in the sequence so far.
#
# The internal structure of an RNN layer - or its variants, the LSTM (long
# short-term memory) and GRU (gated recurrent unit) - is moderately
# complex and beyond the scope of this video, but we’ll show you what one
# looks like in action with an LSTM-based part-of-speech tagger (a type of
# classifier that tells you if a word is a noun, verb, etc.):
#

# 这个模型演示如何用 LSTM 处理一段词序列，并为每个词预测词性标签。
class LSTMTagger(torch.nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super().__init__()
        # hidden_dim 表示 LSTM 隐藏状态的长度，也可以粗略理解为“记忆向量”的大小。
        self.hidden_dim = hidden_dim

        # Embedding 层把词的整数编号转换成稠密向量。
        # 输入是词 ID，输出是长度为 embedding_dim 的向量。
        self.word_embeddings = torch.nn.Embedding(vocab_size, embedding_dim)

        # LSTM 接收词向量序列作为输入，并为序列中的每一步输出 hidden_dim 维隐藏状态。
        # 它适合处理文本、时间序列这类有先后顺序的数据。
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim)

        # 线性层把每个词的隐藏状态映射到标签空间。
        # tagset_size 是标签数量，例如名词、动词、形容词等标签的总数。
        self.hidden2tag = torch.nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        # sentence 中每个元素是一个词在词表里的编号。
        # word_embeddings 会把这些编号变成词向量序列。
        embeds = self.word_embeddings(sentence)
        # LSTM 默认期望输入形状为 [序列长度, batch_size, embedding_dim]。
        # 这里一次只处理 1 个句子，所以 batch_size 是 1。
        # LSTM 返回两个结果：每个时间步的输出，以及最终隐藏状态；这里用 _ 忽略后者。
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        # 把 LSTM 输出整理成 [词数, hidden_dim]，再交给线性分类层。
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        # log_softmax 输出对数概率。训练分类模型时，它常和负对数似然损失一起使用。
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


########################################################################
# The constructor has four arguments:
#
# -  ``vocab_size`` is the number of words in the input vocabulary. Each
#    word is a one-hot vector (or unit vector) in a
#    ``vocab_size``-dimensional space.
# -  ``tagset_size`` is the number of tags in the output set.
# -  ``embedding_dim`` is the size of the *embedding* space for the
#    vocabulary. An embedding maps a vocabulary onto a low-dimensional
#    space, where words with similar meanings are close together in the
#    space.
# -  ``hidden_dim`` is the size of the LSTM’s memory.
#
# The input will be a sentence with the words represented as indices of
# one-hot vectors. The embedding layer will then map these down to an
# ``embedding_dim``-dimensional space. The LSTM takes this sequence of
# embeddings and iterates over it, fielding an output vector of length
# ``hidden_dim``. The final linear layer acts as a classifier; applying
# ``log_softmax()`` to the output of the final layer converts the output
# into a normalized set of estimated probabilities that a given word maps
# to a given tag.
#
# If you’d like to see this network in action, check out the `Sequence
# Models and LSTM
# Networks <https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html>`__
# tutorial on pytorch.org.
#
# Transformers
# ~~~~~~~~~~~~
#
# *Transformers* are multi-purpose networks that have taken over the state
# of the art in NLP with models like BERT. A discussion of transformer
# architecture is beyond the scope of this video, but PyTorch has a
# ``Transformer`` class that allows you to define the overall parameters
# of a transformer model - the number of attention heads, the number of
# encoder & decoder layers, dropout and activation functions, etc. (You
# can even build the BERT model from this single class, with the right
# parameters!) The ``torch.nn.Transformer`` class also has classes to
# encapsulate the individual components (``TransformerEncoder``,
# ``TransformerDecoder``) and subcomponents (``TransformerEncoderLayer``,
# ``TransformerDecoderLayer``). For details, check out the
# `documentation <https://pytorch.org/docs/stable/nn.html#transformer-layers>`__
# on transformer classes.
#
# Other Layers and Functions
# --------------------------
#
# Data Manipulation Layers
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# There are other layer types that perform important functions in models,
# but don’t participate in the learning process themselves.
#
# **Max pooling** (and its twin, min pooling) reduce a tensor by combining
# cells, and assigning the maximum value of the input cells to the output
# cell (we saw this). For example:
#

# 构造一个 1x6x6 的张量。这里可以理解为 1 个通道、大小为 6x6 的特征图。
my_tensor = torch.rand(1, 6, 6)
print(my_tensor)

# MaxPool2d(3) 使用 3x3 窗口做最大池化。
# 对 6x6 输入来说，默认步长等于窗口大小 3，所以输出空间尺寸会变成 2x2。
maxpool_layer = torch.nn.MaxPool2d(3)
print(maxpool_layer(my_tensor))


#########################################################################
# If you look closely at the values above, you’ll see that each of the
# values in the maxpooled output is the maximum value of each quadrant of
# the 6x6 input.
#
# **Normalization layers** re-center and normalize the output of one layer
# before feeding it to another. Centering and scaling the intermediate
# tensors has a number of beneficial effects, such as letting you use
# higher learning rates without exploding/vanishing gradients.
#

# 先生成 [0, 1) 的随机数，再乘以 20、加 5。
# 这样数据大致落在 [5, 25) 之间，均值通常接近 15，方便观察归一化效果。
my_tensor = torch.rand(1, 4, 4) * 20 + 5
print(my_tensor)

# 查看归一化前的整体均值。
print(my_tensor.mean())

# BatchNorm1d(4) 表示有 4 个特征通道。
# 输入形状是 [N, C, L]，这里 N=1、C=4、L=4。
# 它会按通道统计均值和方差，再把数值重新缩放到更稳定的范围。
norm_layer = torch.nn.BatchNorm1d(4)
normed_tensor = norm_layer(my_tensor)
print(normed_tensor)

# 归一化后的均值通常会非常接近 0。
print(normed_tensor.mean())



##########################################################################
# Running the cell above, we’ve added a large scaling factor and offset to
# an input tensor; you should see the input tensor’s ``mean()`` somewhere
# in the neighborhood of 15. After running it through the normalization
# layer, you can see that the values are smaller, and grouped around zero
# - in fact, the mean should be very small (> 1e-8).
#
# This is beneficial because many activation functions (discussed below)
# have their strongest gradients near 0, but sometimes suffer from
# vanishing or exploding gradients for inputs that drive them far away
# from zero. Keeping the data centered around the area of steepest
# gradient will tend to mean faster, better learning and higher feasible
# learning rates.
#
# **Dropout layers** are a tool for encouraging *sparse representations*
# in your model - that is, pushing it to do inference with less data.
#
# Dropout layers work by randomly setting parts of the input tensor
# *during training* - dropout layers are always turned off for inference.
# This forces the model to learn against this masked or reduced dataset.
# For example:
#

# 构造一个示例张量，用来观察 Dropout 随机清零的效果。
my_tensor = torch.rand(1, 4, 4)

# p=0.4 表示训练模式下每个元素有 40% 的概率被置为 0。
# 未被置零的元素会按比例放大，以保持期望值大致不变。
dropout = torch.nn.Dropout(p=0.4)
# 连续调用两次通常会得到不同的置零位置，因为 Dropout 每次都会重新随机采样。
print(dropout(my_tensor))
print(dropout(my_tensor))


##########################################################################
# Above, you can see the effect of dropout on a sample tensor. You can use
# the optional ``p`` argument to set the probability of an individual
# weight dropping out; if you don’t it defaults to 0.5.
#
# Activation Functions
# ~~~~~~~~~~~~~~~~~~~~
#
# Activation functions make deep learning possible. A neural network is
# really a program - with many parameters - that *simulates a mathematical
# function*. If all we did was multiple tensors by layer weights
# repeatedly, we could only simulate *linear functions;* further, there
# would be no point to having many layers, as the whole network would
# reduce could be reduced to a single matrix multiplication. Inserting
# *non-linear* activation functions between layers is what allows a deep
# learning model to simulate any function, rather than just linear ones.
#
# ``torch.nn.Module`` has objects encapsulating all of the major
# activation functions including ReLU and its many variants, Tanh,
# Hardtanh, sigmoid, and more. It also includes other functions, such as
# Softmax, that are most useful at the output stage of a model.
#
# Loss Functions
# ~~~~~~~~~~~~~~
#
# Loss functions tell us how far a model’s prediction is from the correct
# answer. PyTorch contains a variety of loss functions, including common
# MSE (mean squared error = L2 norm), Cross Entropy Loss and Negative
# Likelihood Loss (useful for classifiers), and others.
#
