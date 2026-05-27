"""
中文注释学习版。

原始文件：
beginner_source/introyt/tensors_deeper_tutorial.py

说明：
这个文件基于原教程重新创建，保留主要可执行代码的顺序，并用中文解释关键代码。
它用于学习阅读，不用于替代 PyTorch Tutorials 的正式 Sphinx Gallery 源文件。
"""

# torch 是 PyTorch 的主包，张量创建、数学运算、设备迁移等功能都在这里。
import torch
# math 提供 pi 等数学常量，后面会用它构造角度示例。
import math


# =============================================================================
# 创建张量
# =============================================================================

# empty() 只分配内存，不会把内存里的值清零。
# 所以打印结果可能看起来像随机数，但它们不是 torch.rand() 生成的随机数。
x = torch.empty(3, 4)
print(type(x))  # 查看对象类型，结果是 torch.Tensor。
print(x)  # 查看张量内容；里面的值不可预测。

# zeros() 创建指定形状的全 0 张量。
zeros = torch.zeros(2, 3)
print(zeros)

# ones() 创建指定形状的全 1 张量。
ones = torch.ones(2, 3)
print(ones)

# 固定随机数种子。种子相同，后续随机结果就更容易复现。
torch.manual_seed(1729)
# rand() 创建 [0, 1) 区间内均匀分布的随机浮点张量。
random = torch.rand(2, 3)
print(random)


# =============================================================================
# 随机张量和随机种子
# =============================================================================

# 把随机数生成器重置到同一个起点。
torch.manual_seed(1729)
random1 = torch.rand(2, 3)  # 重置种子后的第 1 次随机生成。
print(random1)

random2 = torch.rand(2, 3)  # 同一随机序列里的第 2 次随机生成。
print(random2)

# 再次使用同一个种子，随机序列会重新开始。
torch.manual_seed(1729)
random3 = torch.rand(2, 3)  # 应该和 random1 相同。
print(random3)

random4 = torch.rand(2, 3)  # 应该和 random2 相同。
print(random4)


# =============================================================================
# 张量形状
# =============================================================================

# 创建一个三维张量，形状是 2 x 2 x 3。
x = torch.empty(2, 2, 3)
print(x.shape)  # shape 记录每个维度的大小。
print(x)

# empty_like() 复制 x 的形状和 dtype，但不初始化具体值。
empty_like_x = torch.empty_like(x)
print(empty_like_x.shape)
print(empty_like_x)

# zeros_like() 创建和 x 形状一样的全 0 张量。
zeros_like_x = torch.zeros_like(x)
print(zeros_like_x.shape)
print(zeros_like_x)

# ones_like() 创建和 x 形状一样的全 1 张量。
ones_like_x = torch.ones_like(x)
print(ones_like_x.shape)
print(ones_like_x)

# rand_like() 创建和 x 形状一样的随机张量。
rand_like_x = torch.rand_like(x)
print(rand_like_x.shape)
print(rand_like_x)


# =============================================================================
# 用 Python 数据创建张量
# =============================================================================

# 嵌套列表会创建二维张量；每个内层列表是一行。
some_constants = torch.tensor([[3.1415926, 2.71828], [1.61803, 0.0072897]])
print(some_constants)

# 一维 tuple 会创建一维张量。
some_integers = torch.tensor((2, 3, 5, 7, 11, 13, 17, 19))
print(some_integers)

# tuple 和 list 可以混用，只要整体形状规则。
more_integers = torch.tensor(((2, 4, 6), [3, 6, 9]))
print(more_integers)


# =============================================================================
# 张量数据类型
# =============================================================================

# dtype 指定张量元素的数据类型。这里创建 int16 整数张量。
a = torch.ones((2, 3), dtype=torch.int16)
print(a)

# 这里创建 float64 随机张量，再乘以 20，让数值范围更明显。
b = torch.rand((2, 3), dtype=torch.float64) * 20.0
print(b)

# to() 可以转换 dtype。浮点数转整数时，小数部分会被截断。
c = b.to(torch.int32)
print(c)


# =============================================================================
# 张量和标量做数学运算
# =============================================================================

# 张量和单个数字做运算时，PyTorch 会把运算应用到每个元素上。
ones = torch.zeros(2, 2) + 1
twos = torch.ones(2, 2) * 2

# 可以像普通 Python 表达式一样连续运算，并遵循正常优先级。
threes = (torch.ones(2, 2) * 7 - 1) / 2
fours = twos**2
sqrt2s = twos**0.5

print(ones)
print(twos)
print(threes)
print(fours)
print(sqrt2s)


# =============================================================================
# 形状相同的张量之间做运算
# =============================================================================

# 两个形状相同的张量做运算时，会按位置逐元素计算。
powers2 = twos ** torch.tensor([[1, 2], [3, 4]])
print(powers2)

# ones 和 fours 都是 2 x 2，所以可以逐元素相加。
fives = ones + fours
print(fives)

# 这里是逐元素乘法，不是矩阵乘法。
dozens = threes * fours
print(dozens)

# 形状不兼容时，下面这种代码会报错，所以这里只作为注释保留。
# a = torch.rand(2, 3)
# b = torch.rand(3, 2)
# print(a * b)


# =============================================================================
# 广播
# =============================================================================

rand = torch.rand(2, 4)

# 右侧张量形状是 1 x 4。广播会把这一行“重复使用”到 rand 的两行上。
doubled = rand * (torch.ones(1, 4) * 2)

print(rand)
print(doubled)

# 创建一个 4 x 3 x 2 的全 1 张量。
a = torch.ones(4, 3, 2)

# torch.rand(3, 2) 可以看成缺少第 0 维，也就是形状类似 1 x 3 x 2。
# 它会沿着第 0 维广播到 a 的 4 层上。
b = a * torch.rand(3, 2)
print(b)

# 最后一维是 1，可以广播成大小为 2 的维度。
c = a * torch.rand(3, 1)
print(c)

# 中间维度是 1，可以广播成大小为 3 的维度。
d = a * torch.rand(1, 2)
print(d)

# 下面这些广播示例会失败，因此只作为注释保留。
# 广播规则通常从最后一个维度往前比较：维度要么相同，要么其中一个是 1，要么某个张量缺少这个维度。
# a = torch.ones(4, 3, 2)
# b = a * torch.rand(4, 3)
# c = a * torch.rand(2, 3)
# d = a * torch.rand((0,))


# =============================================================================
# 更多数学操作
# =============================================================================

# 先构造 [-1, 1) 范围内的随机数，方便观察 abs、ceil、floor、clamp 的效果。
a = torch.rand(2, 4) * 2 - 1
print("Common functions:")
print(torch.abs(a))  # 每个元素取绝对值。
print(torch.ceil(a))  # 每个元素向上取整。
print(torch.floor(a))  # 每个元素向下取整。
print(torch.clamp(a, -0.5, 0.5))  # 把每个元素限制在 [-0.5, 0.5] 范围内。

# 三角函数示例。角度单位是弧度。
angles = torch.tensor([0, math.pi / 4, math.pi / 2, 3 * math.pi / 4])
sines = torch.sin(angles)  # 逐元素计算正弦。
inverses = torch.asin(sines)  # 逐元素计算反正弦。
print("\nSine and arcsine:")
print(angles)
print(sines)
print(inverses)

# 位运算示例。XOR 会按整数的二进制位进行异或。
print("\nBitwise XOR:")
b = torch.tensor([1, 5, 11])
c = torch.tensor([2, 7, 10])
print(torch.bitwise_xor(b, c))

# 比较运算也可以广播，结果是 bool 类型张量。
print("\nBroadcasted, element-wise equality comparison:")
d = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
e = torch.ones(1, 2)
print(torch.eq(d, e))

# 归约操作会把多个元素汇总成更少的结果。
print("\nReduction ops:")
print(torch.max(d))  # 所有元素中的最大值。
print(torch.max(d).item())  # item() 把单元素张量转成 Python 数字。
print(torch.mean(d))  # 平均值。
print(torch.std(d))  # 标准差。
print(torch.prod(d))  # 所有元素的乘积。
print(torch.unique(torch.tensor([1, 2, 1, 2, 1, 2])))  # 去重。

# 向量和线性代数操作。
v1 = torch.tensor([1.0, 0.0, 0.0])  # x 方向单位向量。
v2 = torch.tensor([0.0, 1.0, 0.0])  # y 方向单位向量。
m1 = torch.rand(2, 2)  # 随机矩阵。
m2 = torch.tensor([[3.0, 0.0], [0.0, 3.0]])  # 3 倍单位矩阵。

print("\nVectors & Matrices:")
print(torch.linalg.cross(v2, v1))  # 计算向量叉乘。
print(m1)
m3 = torch.matmul(m1, m2)  # 矩阵乘法，不是逐元素乘法。
print(m3)
print(torch.linalg.svd(m3))  # 奇异值分解。


# =============================================================================
# 原地修改张量
# =============================================================================

a = torch.tensor([0, math.pi / 4, math.pi / 2, 3 * math.pi / 4])
print("a:")
print(a)
print(torch.sin(a))  # 普通 sin() 会创建新张量。
print(a)  # a 没有被改变。

b = torch.tensor([0, math.pi / 4, math.pi / 2, 3 * math.pi / 4])
print("\nb:")
print(b)
print(b.sin_())  # 函数名末尾的下划线表示原地修改。
print(b)  # b 已经变成 sin(b) 的结果。

# 原地加法和乘法示例。
a = torch.ones(2, 2)
b = torch.rand(2, 2)

print("Before:")
print(a)
print(b)
print("\nAfter adding:")
print(a.add_(b))  # 把 a + b 的结果写回 a。
print(a)
print(b)
print("\nAfter multiplying")
print(b.mul_(b))  # 把 b * b 的结果写回 b，相当于逐元素平方。
print(b)


# =============================================================================
# 使用 out 参数复用已有张量
# =============================================================================

a = torch.rand(2, 2)
b = torch.rand(2, 2)
c = torch.zeros(2, 2)
old_id = id(c)  # 记录 c 这个对象的身份。

print(c)
d = torch.matmul(a, b, out=c)  # 把矩阵乘法结果写入已有张量 c。
print(c)

assert c is d  # c 和 d 指向同一个张量对象。
assert id(c) == old_id  # c 的对象身份没有变，说明没有换成新对象。

torch.rand(2, 2, out=c)  # 创建随机数时也可以把结果写入 c。
print(c)
assert id(c) == old_id


# =============================================================================
# 复制张量
# =============================================================================

a = torch.ones(2, 2)
b = a  # b 只是 a 的另一个名字，不是新副本。

a[0][1] = 561  # 修改 a 的一个元素。
print(b)  # b 也会变，因为 a 和 b 指向同一个张量。

# clone() 会复制数据，得到一个真正独立的新张量。
a = torch.ones(2, 2)
b = a.clone()

assert b is not a  # 两者是不同对象。
print(torch.eq(a, b))  # 但内容一开始是相同的。

a[0][1] = 561  # 修改 a。
print(b)  # b 不受影响。


# =============================================================================
# clone()、detach() 和 autograd
# =============================================================================

# requires_grad=True 表示需要记录计算历史，方便之后自动求导。
a = torch.rand(2, 2, requires_grad=True)
print(a)

# clone() 会保留 autograd 关系，b 仍然连接在计算图里。
b = a.clone()
print(b)

# detach() 先切断 autograd 历史，再 clone() 得到不跟踪梯度的副本。
c = a.detach().clone()
print(c)

# detach() 不会改变原来的 a，a 仍然 requires_grad=True。
print(a)


# =============================================================================
# 移动到加速设备
# =============================================================================

# accelerator 可以是 CUDA、MPS、MTIA、XPU 等后端。
if torch.accelerator.is_available():
    print("We have an accelerator!")
else:
    print("Sorry, CPU only.")

# 如果有可用加速器，可以在创建张量时直接指定 device。
if torch.accelerator.is_available():
    gpu_rand = torch.rand(2, 2, device=torch.accelerator.current_accelerator())
    print(gpu_rand)
else:
    print("Sorry, CPU only.")

# 统一保存当前要使用的设备。这样代码可以同时适配 CPU 和加速器。
my_device = (
    torch.accelerator.current_accelerator()
    if torch.accelerator.is_available()
    else torch.device("cpu")
)
print(f"Device: {my_device}")

# 直接在目标设备上创建张量。
x = torch.rand(2, 2, device=my_device)
print(x)

# 也可以先在 CPU 上创建，再用 to() 移动到目标设备。
y = torch.rand(2, 2)
y = y.to(my_device)

# 注意：参与同一次运算的张量必须在同一个设备上。
# 下面这种 CPU 张量和 CUDA 张量直接相加的写法会报错。
# x = torch.rand(2, 2)
# y = torch.rand(2, 2, device="cuda")
# z = x + y


# =============================================================================
# 改变张量形状
# =============================================================================

# 假设这是一张 3 通道、226 x 226 的图片，形状是 C x H x W。
a = torch.rand(3, 226, 226)
# unsqueeze(0) 在最前面加一个大小为 1 的维度，变成 N x C x H x W。
# 这常用于把单张图片变成 batch 大小为 1 的输入。
b = a.unsqueeze(0)

print(a.shape)
print(b.shape)

# 这个张量有很多大小为 1 的维度。
c = torch.rand(1, 1, 1, 1, 1)
print(c)

# 模型输出可能是形状 (1, 20)，表示 batch 中 1 个样本，每个样本 20 个数。
a = torch.rand(1, 20)
print(a.shape)
print(a)

# squeeze(0) 删除第 0 维；只有这个维度大小为 1 时才能删除。
b = a.squeeze(0)
print(b.shape)
print(b)

c = torch.rand(2, 2)
print(c.shape)

# 第 0 维大小是 2，不是 1，所以 squeeze(0) 不会改变形状。
d = c.squeeze(0)
print(d.shape)

# unsqueeze() 也常用于让张量满足广播规则。
a = torch.ones(4, 3, 2)
b = torch.rand(3)  # 如果直接 a * b，会因为最后维度 2 和 3 不匹配而报错。
# 把 b 从形状 (3,) 改成 (3, 1)，这样就能和 a 的后两维 (3, 2) 广播。
c = b.unsqueeze(1)
print(c.shape)
print(a * c)

# 带下划线的 unsqueeze_() 会直接修改原张量。
batch_me = torch.rand(3, 226, 226)
print(batch_me.shape)
batch_me.unsqueeze_(0)
print(batch_me.shape)


# =============================================================================
# reshape
# =============================================================================

# 模拟卷积层输出：6 个特征图，每个特征图 20 x 20。
output3d = torch.rand(6, 20, 20)
print(output3d.shape)

# reshape() 把 3 维张量拉平成 1 维向量。
# 新形状的元素总数必须等于原来的元素总数，也就是 6 * 20 * 20。
input1d = output3d.reshape(6 * 20 * 20)
print(input1d.shape)

# reshape 也可以用 torch.reshape() 的函数形式调用。
# (6 * 20 * 20,) 是只有一个元素的 tuple，末尾逗号不能省。
print(torch.reshape(output3d, (6 * 20 * 20,)).shape)


# =============================================================================
# NumPy 和 PyTorch 互转
# =============================================================================

import numpy as np

# 创建一个 2 x 3 的 NumPy 数组，元素全是 1。
numpy_array = np.ones((2, 3))
print(numpy_array)

# from_numpy() 会基于同一块内存创建 PyTorch 张量。
pytorch_tensor = torch.from_numpy(numpy_array)
print(pytorch_tensor)

# 创建一个普通的 CPU 张量。
pytorch_rand = torch.rand(2, 3)
print(pytorch_rand)

# numpy() 会把 CPU 张量暴露成 NumPy 数组。
numpy_rand = pytorch_rand.numpy()
print(numpy_rand)

# 因为 from_numpy() 共享内存，所以修改 NumPy 数组会影响 PyTorch 张量。
numpy_array[1, 1] = 23
print(pytorch_tensor)

# 反过来也一样：修改 PyTorch 张量会影响对应的 NumPy 数组。
pytorch_rand[1, 1] = 17
print(numpy_rand)
