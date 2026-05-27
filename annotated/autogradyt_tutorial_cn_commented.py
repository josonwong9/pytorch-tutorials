"""
`Introduction <introyt1_tutorial.html>`_ ||
`Tensors <tensors_deeper_tutorial.html>`_ ||
**Autograd** ||
`Building Models <modelsyt_tutorial.html>`_ ||
`TensorBoard Support <tensorboardyt_tutorial.html>`_ ||
`Training Models <trainingyt.html>`_ ||
`Model Understanding <captumyt.html>`_

The Fundamentals of Autograd
============================

Follow along with the video below or on `youtube <https://www.youtube.com/watch?v=M0fX15_-xrY>`__.

.. raw:: html

   <div style="margin-top:10px; margin-bottom:10px;">
     <iframe width="560" height="315" src="https://www.youtube.com/embed/M0fX15_-xrY" frameborder="0" allow="accelerometer; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
   </div>

.. note::

   这是 ``autogradyt_tutorial.py`` 的中文注释副本。原教程的英文讲解和
   PyTorch 示例代码保持不变，新增的中文注释会解释每段代码在做什么、
   为什么这样写，以及这些操作和自动求导之间的关系。

PyTorch’s *Autograd* feature is part of what make PyTorch flexible and
fast for building machine learning projects. It allows for the rapid and
easy computation of multiple partial derivatives (also referred to as
*gradients)* over a complex computation. This operation is central to
backpropagation-based neural network learning.

The power of autograd comes from the fact that it traces your
computation dynamically *at runtime,* meaning that if your model has
decision branches, or loops whose lengths are not known until runtime,
the computation will still be traced correctly, and you’ll get correct
gradients to drive learning. This, combined with the fact that your
models are built in Python, offers far more flexibility than frameworks
that rely on static analysis of a more rigidly-structured model for
computing gradients.

What Do We Need Autograd For?
-----------------------------

"""

###########################################################################
# 中文理解：
# 训练神经网络时，我们希望模型预测值尽量接近真实目标值。二者的差距由
# loss（损失）表示。为了让 loss 变小，需要知道每个可学习参数应该往哪个
# 方向调整。这个“方向”和“幅度”就是梯度。
#
# autograd 的作用，就是在你执行普通 PyTorch 代码时自动记录计算过程，
# 然后在 backward() 时根据链式法则把梯度从 loss 一路传回各个参数。
#
# A machine learning model is a *function*, with inputs and outputs. For
# this discussion, we’ll treat the inputs as an *i*-dimensional vector
# :math:`\vec{x}`, with elements :math:`x_{i}`. We can then express the
# model, *M*, as a vector-valued function of the input: :math:`\vec{y} =
# \vec{M}(\vec{x})`. (We treat the value of M’s output as
# a vector because in general, a model may have any number of outputs.)
#
# Since we’ll mostly be discussing autograd in the context of training,
# our output of interest will be the model’s loss. The *loss function*
# L(:math:`\vec{y}`) = L(:math:`\vec{M}`\ (:math:`\vec{x}`)) is a
# single-valued scalar function of the model’s output. This function
# expresses how far off our model’s prediction was from a particular
# input’s *ideal* output. *Note: After this point, we will often omit the
# vector sign where it should be contextually clear - e.g.,* :math:`y`
# instead of :math:`\vec y`.
#
# In training a model, we want to minimize the loss. In the idealized case
# of a perfect model, that means adjusting its learning weights - that is,
# the adjustable parameters of the function - such that loss is zero for
# all inputs. In the real world, it means an iterative process of nudging
# the learning weights until we see that we get a tolerable loss for a
# wide variety of inputs.
#
# How do we decide how far and in which direction to nudge the weights? We
# want to *minimize* the loss, which means making its first derivative
# with respect to the input equal to 0:
# :math:`\frac{\partial L}{\partial x} = 0`.
#
# Recall, though, that the loss is not *directly* derived from the input,
# but a function of the model’s output (which is a function of the input
# directly), :math:`\frac{\partial L}{\partial x}` =
# :math:`\frac{\partial {L({\vec y})}}{\partial x}`. By the chain rule of
# differential calculus, we have
# :math:`\frac{\partial {L({\vec y})}}{\partial x}` =
# :math:`\frac{\partial L}{\partial y}\frac{\partial y}{\partial x}` =
# :math:`\frac{\partial L}{\partial y}\frac{\partial M(x)}{\partial x}`.
#
# :math:`\frac{\partial M(x)}{\partial x}` is where things get complex.
# The partial derivatives of the model’s outputs with respect to its
# inputs, if we were to expand the expression using the chain rule again,
# would involve many local partial derivatives over every multiplied
# learning weight, every activation function, and every other mathematical
# transformation in the model. The full expression for each such partial
# derivative is the sum of the products of the local gradient of *every
# possible path* through the computation graph that ends with the variable
# whose gradient we are trying to measure.
#
# In particular, the gradients over the learning weights are of interest
# to us - they tell us *what direction to change each weight* to get the
# loss function closer to zero.
#
# Since the number of such local derivatives (each corresponding to a
# separate path through the model’s computation graph) will tend to go up
# exponentially with the depth of a neural network, so does the complexity
# in computing them. This is where autograd comes in: It tracks the
# history of every computation. Every computed tensor in your PyTorch
# model carries a history of its input tensors and the function used to
# create it. Combined with the fact that PyTorch functions meant to act on
# tensors each have a built-in implementation for computing their own
# derivatives, this greatly speeds the computation of the local
# derivatives needed for learning.
#
# A Simple Example
# ----------------
#
# That was a lot of theory - but what does it look like to use autograd in
# practice?
#
# Let’s start with a straightforward example. First, we’ll do some imports
# to let us graph our results:
#

# %matplotlib inline

# 导入 PyTorch。后续所有张量、神经网络层、优化器和自动求导功能都来自 torch。
import torch

# matplotlib 用来画图，帮助我们直观看到函数和梯度的形状。
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import math

#########################################################################
# Next, we’ll create an input tensor full of evenly spaced values on the
# interval :math:`[0, 2{\pi}]`, and specify ``requires_grad=True``. (Like
# most functions that create tensors, ``torch.linspace()`` accepts an
# optional ``requires_grad`` option.) Setting this flag means that in
# every computation that follows, autograd will be accumulating the
# history of the computation in the output tensors of that computation.
#

a = torch.linspace(0.0, 2.0 * math.pi, steps=25, requires_grad=True)
# 这里的 a 是 25 个从 0 到 2π 均匀分布的点。
# requires_grad=True 表示：凡是由 a 参与产生的新张量，PyTorch 都会记录计算历史。
# 这些历史会在之后调用 backward() 时用来按链式法则计算梯度。
print(a)


########################################################################
# Next, we’ll perform a computation, and plot its output in terms of its
# inputs:
#

b = torch.sin(a)
# b 保存 sin(a) 的结果。因为 a 开启了 requires_grad，b 会带有 grad_fn。
# grad_fn 可以理解为“b 是怎么被算出来的”，这里对应正弦函数的反向传播规则。
plt.plot(a.detach(), b.detach())
plt.show()
# 画图时使用 detach()，是因为 matplotlib 不需要、也不能直接处理带梯度历史的张量。
# detach() 会返回一个不再连接计算图的新视图，适合用于打印、画图等观察操作。


########################################################################
# Let’s have a closer look at the tensor ``b``. When we print it, we see
# an indicator that it is tracking its computation history:
#

print(b)


#######################################################################
# This ``grad_fn`` gives us a hint that when we execute the
# backpropagation step and compute gradients, we’ll need to compute the
# derivative of :math:`\sin(x)` for all this tensor’s inputs.
#
# Let’s perform some more computations:
#

c = 2 * b
# c = 2 * sin(a)。乘以常数 2 会让最终梯度也乘以 2。
print(c)

d = c + 1
# d = 2 * sin(a) + 1。加常数只会整体上移函数值，不改变它对 a 的导数。
print(d)


##########################################################################
# Finally, let’s compute a single-element output. When you call
# ``.backward()`` on a tensor with no arguments, it expects the calling
# tensor to contain only a single element, as is the case when computing a
# loss function.
#

out = d.sum()
# backward() 默认只能直接作用在标量输出上。
# d 是一个包含 25 个元素的向量，所以这里先用 sum() 把它汇总成单个标量 out。
print(out)


##########################################################################
# Each ``grad_fn`` stored with our tensors allows you to walk the
# computation all the way back to its inputs with its ``next_functions``
# property. We can see below that drilling down on this property on ``d``
# shows us the gradient functions for all the prior tensors. Note that
# ``a.grad_fn`` is reported as ``None``, indicating that this was an input
# to the function with no history of its own.
#

print("d:")
print(d.grad_fn)
print(d.grad_fn.next_functions)
print(d.grad_fn.next_functions[0][0].next_functions)
print(d.grad_fn.next_functions[0][0].next_functions[0][0].next_functions)
print(
    d.grad_fn.next_functions[0][0]
    .next_functions[0][0]
    .next_functions[0][0]
    .next_functions
)
print("\nc:")
print(c.grad_fn)
print("\nb:")
print(b.grad_fn)
print("\na:")
print(a.grad_fn)
# a 是用户直接创建的叶子张量，不是由其他张量计算出来的，所以 a.grad_fn 是 None。
# 但因为 a.requires_grad=True，反向传播后它的梯度会保存在 a.grad 中。


######################################################################
# With all this machinery in place, how do we get derivatives out? You
# call the ``backward()`` method on the output, and check the input’s
# ``grad`` property to inspect the gradients:
#

out.backward()
# 从标量 out 开始反向传播，PyTorch 会沿着刚才记录的计算图一路往回走。
# 这个过程会把 out 对叶子张量 a 的导数累积到 a.grad。
print(a.grad)
plt.plot(a.detach(), a.grad.detach())
plt.show()
# 按前面的计算，out = sum(2 * sin(a) + 1)，所以对每个 a 的梯度应为 2 * cos(a)。


#########################################################################
# Recall the computation steps we took to get here:
#
# .. code-block:: python
#
#    a = torch.linspace(0., 2. * math.pi, steps=25, requires_grad=True)
#    b = torch.sin(a)
#    c = 2 * b
#    d = c + 1
#    out = d.sum()
#
# Adding a constant, as we did to compute ``d``, does not change the
# derivative. That leaves :math:`c = 2 * b = 2 * \sin(a)`, the derivative
# of which should be :math:`2 * \cos(a)`. Looking at the graph above,
# that’s just what we see.
#
# Be aware that only *leaf nodes* of the computation have their gradients
# computed. If you tried, for example, ``print(c.grad)`` you’d get back
# ``None``. In this simple example, only the input is a leaf node, so only
# it has gradients computed.
#
# Autograd in Training
# --------------------
#
# 中文理解：
# 真实训练通常包含四步：前向传播得到 prediction，计算 loss，执行
# loss.backward() 得到梯度，最后用 optimizer.step() 更新参数。下面的
# TinyModel 示例会把这四步拆开观察。
#
# We’ve had a brief look at how autograd works, but how does it look when
# it’s used for its intended purpose? Let’s define a small model and
# examine how it changes after a single training batch. First, define a
# few constants, our model, and some stand-ins for inputs and outputs:
#

BATCH_SIZE = 16
DIM_IN = 1000
HIDDEN_SIZE = 100
DIM_OUT = 10
# 这四个常量描述一个很小的全连接网络：
# - 每个 batch 有 16 条样本。
# - 每条输入样本有 1000 个特征。
# - 中间隐藏层有 100 个神经元。
# - 输出层有 10 个值，常见于 10 类分类任务的结构。


class TinyModel(torch.nn.Module):
    # 自定义模型通常继承 torch.nn.Module。
    # Module 会自动管理层、参数、设备移动、训练/评估模式等常见功能。

    def __init__(self):
        super(TinyModel, self).__init__()

        self.layer1 = torch.nn.Linear(DIM_IN, HIDDEN_SIZE)
        # 第一层线性层：把 1000 维输入映射到 100 维隐藏表示。
        # Linear 层内部有 weight 和 bias，它们默认 requires_grad=True。
        self.relu = torch.nn.ReLU()
        # ReLU 是非线性激活函数，把负数截断为 0，让模型能拟合非线性关系。
        self.layer2 = torch.nn.Linear(HIDDEN_SIZE, DIM_OUT)
        # 第二层线性层：把隐藏表示映射到 10 维输出。

    def forward(self, x):
        # forward 定义一次前向计算如何从输入 x 得到输出。
        # 调用 model(some_input) 时，PyTorch 会自动执行这个方法。
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x


some_input = torch.randn(BATCH_SIZE, DIM_IN, requires_grad=False)
ideal_output = torch.randn(BATCH_SIZE, DIM_OUT, requires_grad=False)
# 这里用随机数模拟一批输入和“理想输出”。
# 数据本身通常不需要求梯度，所以 requires_grad=False。
# 训练时真正需要更新的是模型参数，而不是输入样本。

model = TinyModel()
# 实例化模型后，layer1 和 layer2 的参数已经存在，并会被 autograd 跟踪。


##########################################################################
# One thing you might notice is that we never specify
# ``requires_grad=True`` for the model’s layers. Within a subclass of
# ``torch.nn.Module``, it’s assumed that we want to track gradients on the
# layers’ weights for learning.
#
# If we look at the layers of the model, we can examine the values of the
# weights, and verify that no gradients have been computed yet:
#

print(model.layer2.weight[0][0:10])  # 只打印一小段权重，方便观察数值变化。
print(model.layer2.weight.grad)
# 训练开始前还没有执行 backward()，所以参数的 grad 通常是 None。


##########################################################################
# Let’s see how this changes when we run through one training batch. For a
# loss function, we’ll just use the square of the Euclidean distance
# between our ``prediction`` and the ``ideal_output``, and we’ll use a
# basic stochastic gradient descent optimizer.
#

optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
# optimizer 拿到 model.parameters() 后，就知道哪些参数需要被更新。
# lr 是学习率，表示每次根据梯度调整参数时步子有多大。

prediction = model(some_input)
# 前向传播：输入 some_input 经过 TinyModel，得到当前参数下的预测值。

loss = (ideal_output - prediction).pow(2).sum()
# 这是一个简单的平方误差损失：预测值和目标值越接近，loss 越小。
# sum() 把所有元素汇总成一个标量，方便直接调用 loss.backward()。
print(loss)


######################################################################
# Now, let’s call ``loss.backward()`` and see what happens:
#

loss.backward()
# 反向传播后，每个可学习参数的 .grad 中都会累积 loss 对该参数的梯度。
print(model.layer2.weight[0][0:10])
print(model.layer2.weight.grad[0][0:10])
# 注意：这里权重值还没有变，只是 grad 已经被计算出来。


########################################################################
# We can see that the gradients have been computed for each learning
# weight, but the weights remain unchanged, because we haven’t run the
# optimizer yet. The optimizer is responsible for updating model weights
# based on the computed gradients.
#

optimizer.step()
# optimizer.step() 才会真正根据参数的 grad 更新权重。
# 对 SGD 来说，大致就是 weight = weight - lr * weight.grad。
print(model.layer2.weight[0][0:10])
print(model.layer2.weight.grad[0][0:10])


######################################################################
# You should see that ``layer2``\ ’s weights have changed.
#
# One important thing about the process: After calling
# ``optimizer.step()``, you need to call ``optimizer.zero_grad()``, or
# else every time you run ``loss.backward()``, the gradients on the
# learning weights will accumulate:
#

print(model.layer2.weight.grad[0][0:10])

for i in range(0, 5):
    # 这里故意连续做 5 次 backward()，但中间没有清空梯度。
    # PyTorch 的梯度默认是“累加”的，不会在每次 backward() 前自动归零。
    prediction = model(some_input)
    loss = (ideal_output - prediction).pow(2).sum()
    loss.backward()

print(model.layer2.weight.grad[0][0:10])

optimizer.zero_grad(set_to_none=False)
# 清空梯度是标准训练循环中的关键步骤。
# set_to_none=False 表示把梯度张量清零，而不是把 grad 字段设回 None。

print(model.layer2.weight.grad[0][0:10])


#########################################################################
# After running the cell above, you should see that after running
# ``loss.backward()`` multiple times, the magnitudes of most of the
# gradients will be much larger. Failing to zero the gradients before
# running your next training batch will cause the gradients to blow up in
# this manner, causing incorrect and unpredictable learning results.
#
# Turning Autograd Off and On
# ---------------------------
#
# 中文理解：
# 不是所有计算都需要梯度。例如验证模型、做推理、画图、记录日志时，只关心
# 数值结果，不需要保存计算图。关闭 autograd 可以减少内存占用，也能避免
# 意外把无关计算接入训练图。
#
# There are situations where you will need fine-grained control over
# whether autograd is enabled. There are multiple ways to do this,
# depending on the situation.
#
# The simplest is to change the ``requires_grad`` flag on a tensor
# directly:
#

a = torch.ones(2, 3, requires_grad=True)
# 创建一个 2 行 3 列、元素全为 1 的张量，并开启梯度跟踪。
print(a)

b1 = 2 * a
# 因为 a 正在被 autograd 跟踪，所以 b1 会记录“由 a 乘以 2 得到”的计算历史。
print(b1)

a.requires_grad = False
# 直接把 a 的 requires_grad 改成 False。
# 从这一行之后，用 a 参与的新计算不会继续记录梯度历史。
b2 = 2 * a
print(b2)


##########################################################################
# In the cell above, we see that ``b1`` has a ``grad_fn`` (i.e., a traced
# computation history), which is what we expect, since it was derived from
# a tensor, ``a``, that had autograd turned on. When we turn off autograd
# explicitly with ``a.requires_grad = False``, computation history is no
# longer tracked, as we see when we compute ``b2``.
#
# If you only need autograd turned off temporarily, a better way is to use
# the ``torch.no_grad()``:
#

a = torch.ones(2, 3, requires_grad=True) * 2
b = torch.ones(2, 3, requires_grad=True) * 3
# a 和 b 都来自开启梯度跟踪的张量，所以默认会参与计算图。

c1 = a + b
# 普通上下文中执行加法，c1 会有 grad_fn。
print(c1)

with torch.no_grad():
    # no_grad() 的作用范围只在这个 with 代码块内。
    # 它常用于推理、评估、打印中间结果等不需要训练梯度的场景。
    c2 = a + b

print(c2)

c3 = a * b
# 离开 no_grad() 后，autograd 又恢复正常跟踪，所以 c3 会重新带有 grad_fn。
print(c3)


##########################################################################
# ``torch.no_grad()`` can also be used as a function or method decorator:
#


def add_tensors1(x, y):
    # 普通函数：如果输入需要梯度，输出也会继续记录计算历史。
    return x + y


@torch.no_grad()
def add_tensors2(x, y):
    # 装饰器写法等价于在整个函数内部包一层 with torch.no_grad()。
    # 适合标记“这个函数永远不参与梯度计算”。
    return x + y


a = torch.ones(2, 3, requires_grad=True) * 2
b = torch.ones(2, 3, requires_grad=True) * 3

c1 = add_tensors1(a, b)
print(c1)

c2 = add_tensors2(a, b)
print(c2)
# 对比 c1 和 c2 的打印结果，可以看到 c2 不再带 grad_fn。


##########################################################################
# There’s a corresponding context manager, ``torch.enable_grad()``, for
# turning autograd on when it isn’t already. It may also be used as a
# decorator.
#
# Finally, you may have a tensor that requires gradient tracking, but you
# want a copy that does not. For this we have the ``Tensor`` object’s
# ``detach()`` method - it creates a copy of the tensor that is *detached*
# from the computation history:
#

x = torch.rand(5, requires_grad=True)
y = x.detach()
# y 和 x 共享当前数值，但 y 已经从计算图中分离。
# 后续如果只想拿张量的值做展示、保存或转 NumPy，detach() 很常用。

print(x)
print(y)


#########################################################################
# We did this above when we wanted to graph some of our tensors. This is
# because ``matplotlib`` expects a NumPy array as input, and the implicit
# conversion from a PyTorch tensor to a NumPy array is not enabled for
# tensors with requires_grad=True. Making a detached copy lets us move
# forward.
#
# Autograd and In-place Operations
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# In every example in this notebook so far, we’ve used variables to
# capture the intermediate values of a computation. Autograd needs these
# intermediate values to perform gradient computations. *For this reason,
# you must be careful about using in-place operations when using
# autograd.* Doing so can destroy information you need to compute
# derivatives in the ``backward()`` call. PyTorch will even stop you if
# you attempt an in-place operation on leaf variable that requires
# autograd, as shown below.
#
# .. note::
#     The following code cell throws a runtime error. This is expected.
#
#    .. code-block:: python
#
#       a = torch.linspace(0., 2. * math.pi, steps=25, requires_grad=True)
#       a.sin_()
#

#########################################################################
# Autograd Profiler
# -----------------
#
# Autograd tracks every step of your computation in detail. Such a
# computation history, combined with timing information, would make a
# handy profiler - and autograd has that feature baked in. Here’s a quick
# example usage:
#

device = torch.device("cpu")
run_on_gpu = False
if torch.cuda.is_available():
    device = torch.device("cuda")
    run_on_gpu = True
# 如果当前机器有 CUDA，就让 profiler 记录 CUDA 相关信息；否则只记录 CPU。
# 这里的 device 变量保留了运行设备，run_on_gpu 会传给 profiler。

x = torch.randn(2, 3, requires_grad=True)
y = torch.rand(2, 3, requires_grad=True)
z = torch.ones(2, 3, requires_grad=True)
# 准备三个会被 autograd 跟踪的小张量，用来制造一段可分析的计算。

with torch.autograd.profiler.profile(use_cuda=run_on_gpu) as prf:
    # profiler 会记录 with 块内部每个 autograd 操作的耗时等信息。
    for _ in range(1000):
        z = (z / x) * y
        # 反复执行除法和乘法，让 profiler 有足够的操作可以统计。

print(prf.key_averages().table(sort_by="self_cpu_time_total"))
# key_averages() 会把同类操作聚合起来，table() 以表格形式输出耗时摘要。


##########################################################################
# The profiler can also label individual sub-blocks of code, break out the
# data by input tensor shape, and export data as a Chrome tracing tools
# file. For full details of the API, see the
# `documentation <https://pytorch.org/docs/stable/autograd.html#profiler>`__.
#
# Advanced Topic: More Autograd Detail and the High-Level API
# -----------------------------------------------------------
#
# 中文理解：
# 普通 backward() 最常见，因为训练神经网络时通常只需要 loss 对参数的梯度。
# 但有些研究或数值计算场景需要完整 Jacobian、Hessian，或者显式计算
# vector-Jacobian product。torch.autograd.functional 提供了这些高阶接口。
#
# If you have a function with an n-dimensional input and m-dimensional
# output, :math:`\vec{y}=f(\vec{x})`, the complete gradient is a matrix of
# the derivative of every output with respect to every input, called the
# *Jacobian:*
#
# .. math::
#
#      J
#      =
#      \left(\begin{array}{ccc}
#      \frac{\partial y_{1}}{\partial x_{1}} & \cdots & \frac{\partial y_{1}}{\partial x_{n}}\\
#      \vdots & \ddots & \vdots\\
#      \frac{\partial y_{m}}{\partial x_{1}} & \cdots & \frac{\partial y_{m}}{\partial x_{n}}
#      \end{array}\right)
#
# If you have a second function, :math:`l=g\left(\vec{y}\right)` that
# takes m-dimensional input (that is, the same dimensionality as the
# output above), and returns a scalar output, you can express its
# gradients with respect to :math:`\vec{y}` as a column vector,
# :math:`v=\left(\begin{array}{ccc}\frac{\partial l}{\partial y_{1}} & \cdots & \frac{\partial l}{\partial y_{m}}\end{array}\right)^{T}`
# - which is really just a one-column Jacobian.
#
# More concretely, imagine the first function as your PyTorch model (with
# potentially many inputs and many outputs) and the second function as a
# loss function (with the model’s output as input, and the loss value as
# the scalar output).
#
# If we multiply the first function’s Jacobian by the gradient of the
# second function, and apply the chain rule, we get:
#
# .. math::
#
#    J^{T}\cdot v=\left(\begin{array}{ccc}
#    \frac{\partial y_{1}}{\partial x_{1}} & \cdots & \frac{\partial y_{m}}{\partial x_{1}}\\
#    \vdots & \ddots & \vdots\\
#    \frac{\partial y_{1}}{\partial x_{n}} & \cdots & \frac{\partial y_{m}}{\partial x_{n}}
#    \end{array}\right)\left(\begin{array}{c}
#    \frac{\partial l}{\partial y_{1}}\\
#    \vdots\\
#    \frac{\partial l}{\partial y_{m}}
#    \end{array}\right)=\left(\begin{array}{c}
#    \frac{\partial l}{\partial x_{1}}\\
#    \vdots\\
#    \frac{\partial l}{\partial x_{n}}
#    \end{array}\right)
#
# Note: You could also use the equivalent operation :math:`v^{T}\cdot J`,
# and get back a row vector.
#
# The resulting column vector is the *gradient of the second function with
# respect to the inputs of the first* - or in the case of our model and
# loss function, the gradient of the loss with respect to the model
# inputs.
#
# **``torch.autograd`` is an engine for computing these products.** This
# is how we accumulate the gradients over the learning weights during the
# backward pass.
#
# For this reason, the ``backward()`` call can *also* take an optional
# vector input. This vector represents a set of gradients over the tensor,
# which are multiplied by the Jacobian of the autograd-traced tensor that
# precedes it. Let’s try a specific example with a small vector:
#

x = torch.randn(3, requires_grad=True)
# 这里 x 是长度为 3 的向量，而不是标量。
# 对向量输出做反向传播时，需要额外告诉 autograd 每个输出分量的上游梯度。

y = x * 2
while y.data.norm() < 1000:
    # 不断把 y 翻倍，直到它的范数足够大。
    # 这会构造一条较长但很简单的计算链，便于观察梯度如何随翻倍操作放大。
    y = y * 2

print(y)


##########################################################################
# If we tried to call ``y.backward()`` now, we’d get a runtime error and a
# message that gradients can only be *implicitly* computed for scalar
# outputs. For a multi-dimensional output, autograd expects us to provide
# gradients for those three outputs that it can multiply into the
# Jacobian:
#

v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)  # 用来代表来自后续计算的上游梯度。
# y 有 3 个元素，所以传入 backward() 的 v 也必须有 3 个元素。
# 直观理解：v 告诉 autograd“最终目标对 y 的每个分量有多敏感”。
y.backward(v)

print(x.grad)
# x.grad 保存的是向量-Jacobian 乘积的结果，也就是 v 通过 y 关于 x 的导数传回 x 后的梯度。


##########################################################################
# (Note that the output gradients are all related to powers of two - which
# we’d expect from a repeated doubling operation.)
#
# The High-Level API
# ~~~~~~~~~~~~~~~~~~
#
# There is an API on autograd that gives you direct access to important
# differential matrix and vector operations. In particular, it allows you
# to calculate the Jacobian and the *Hessian* matrices of a particular
# function for particular inputs. (The Hessian is like the Jacobian, but
# expresses all partial *second* derivatives.) It also provides methods
# for taking vector products with these matrices.
#
# Let’s take the Jacobian of a simple function, evaluated for a 2
# single-element inputs:
#


def exp_adder(x, y):
    # 一个小函数：第一项是 2 * e^x，第二项是 3 * y。
    # 它对 x 的导数是 2 * e^x，对 y 的导数是 3。
    return 2 * x.exp() + 3 * y


inputs = (torch.rand(1), torch.rand(1))  # arguments for the function
print(inputs)
torch.autograd.functional.jacobian(exp_adder, inputs)
# jacobian() 会直接计算函数输出对每个输入的偏导数。
# 这里输入是两个单元素张量，所以结果也很小，适合手动检查。


########################################################################
# If you look closely, the first output should equal :math:`2e^x` (since
# the derivative of :math:`e^x` is :math:`e^x`), and the second value
# should be 3.
#
# You can, of course, do this with higher-order tensors:
#

inputs = (torch.rand(3), torch.rand(3))  # arguments for the function
print(inputs)
torch.autograd.functional.jacobian(exp_adder, inputs)
# 当输入是更高维张量时，Jacobian 的形状会包含“输出形状 + 输入形状”。
# 这就是为什么高维 Jacobian 往往比普通梯度更大、更难手算。


#########################################################################
# The ``torch.autograd.functional.hessian()`` method works identically
# (assuming your function is twice differentiable), but returns a matrix
# of all second derivatives.
#
# There is also a function to directly compute the vector-Jacobian
# product, if you provide the vector:
#


def do_some_doubling(x):
    y = x * 2
    while y.data.norm() < 1000:
        # 和前面的例子一样，不断翻倍来构造一个可求导的向量函数。
        y = y * 2
    return y


inputs = torch.randn(3)
my_gradients = torch.tensor([0.1, 1.0, 0.0001])
torch.autograd.functional.vjp(do_some_doubling, inputs, v=my_gradients)
# vjp 是 vector-Jacobian product，即“向量乘以 Jacobian”。
# 它常用于反向模式自动微分，也是神经网络反向传播中最核心的计算形式之一。


##############################################################################
# The ``torch.autograd.functional.jvp()`` method performs the same matrix
# multiplication as ``vjp()`` with the operands reversed. The ``vhp()``
# and ``hvp()`` methods do the same for a vector-Hessian product.
#
# For more information, including performance notes on the `docs for the
# functional
# API <https://pytorch.org/docs/stable/autograd.html#functional-higher-level-api>`__
#
