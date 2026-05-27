"""
`Introduction <introyt1_tutorial.html>`_ ||
`Tensors <tensors_deeper_tutorial.html>`_ ||
`Autograd <autogradyt_tutorial.html>`_ ||
`Building Models <modelsyt_tutorial.html>`_ ||
`TensorBoard Support <tensorboardyt_tutorial.html>`_ ||
`Training Models <trainingyt.html>`_ ||
**Model Understanding**

Model Understanding with Captum
===============================

中文注释版说明：
本文件基于 ``beginner_source/introyt/captumyt.py`` 重新创建。
原教程的英文说明、代码顺序和运行逻辑保持不变；新增的中文注释用于帮助初学者
理解 Captum 如何解释模型预测，以及 Integrated Gradients、Occlusion 和
Layer GradCAM 三种归因方法各自关注什么。

Follow along with the video below or on `youtube <https://www.youtube.com/watch?v=Am2EF9CLu-g>`__. Download the notebook and corresponding files
`here <https://pytorch-tutorial-assets.s3.amazonaws.com/youtube-series/video7.zip>`__.

.. raw:: html

   <div style="margin-top:10px; margin-bottom:10px;">
     <iframe width="560" height="315" src="https://www.youtube.com/embed/Am2EF9CLu-g" frameborder="0" allow="accelerometer; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
   </div>

`Captum <https://captum.ai/>`__ (“comprehension” in Latin) is an open
source, extensible library for model interpretability built on PyTorch.

With the increase in model complexity and the resulting lack of
transparency, model interpretability methods have become increasingly
important. Model understanding is both an active area of research as
well as an area of focus for practical applications across industries
using machine learning. Captum provides state-of-the-art algorithms,
including Integrated Gradients, to provide researchers and developers
with an easy way to understand which features are contributing to a
model’s output.

Full documentation, an API reference, and a suite of tutorials on
specific topics are available at the `captum.ai <https://captum.ai/>`__
website.

Introduction
------------

Captum’s approach to model interpretability is in terms of
*attributions.* There are three kinds of attributions available in
Captum:

-  **Feature Attribution** seeks to explain a particular output in terms
   of features of the input that generated it. Explaining whether a
   movie review was positive or negative in terms of certain words in
   the review is an example of feature attribution.
-  **Layer Attribution** examines the activity of a model’s hidden layer
   subsequent to a particular input. Examining the spatially-mapped
   output of a convolutional layer in response to an input image in an
   example of layer attribution.
-  **Neuron Attribution** is analagous to layer attribution, but focuses
   on the activity of a single neuron.

In this interactive notebook, we’ll look at Feature Attribution and
Layer Attribution.

Each of the three attribution types has multiple **attribution
algorithms** associated with it. Many attribution algorithms fall into
two broad categories:

-  **Gradient-based algorithms** calculate the backward gradients of a
   model output, layer output, or neuron activation with respect to the
   input. **Integrated Gradients** (for features), **Layer Gradient \\*
   Activation**, and **Neuron Conductance** are all gradient-based
   algorithms.
-  **Perturbation-based algorithms** examine the changes in the output
   of a model, layer, or neuron in response to changes in the input. The
   input perturbations may be directed or random. **Occlusion,**
   **Feature Ablation,** and **Feature Permutation** are all
   perturbation-based algorithms.

We’ll be examining algorithms of both types below.

Especially where large models are involved, it can be valuable to
visualize attribution data in ways that relate it easily to the input
features being examined. While it is certainly possible to create your
own visualizations with Matplotlib, Plotly, or similar tools, Captum
offers enhanced tools specific to its attributions:

-  The ``captum.attr.visualization`` module (imported below as ``viz``)
   provides helpful functions for visualizing attributions related to
   images.

This visualization toolset will be demonstrated throughout this notebook.

Installation
------------

Before you get started, you need to have a Python environment with:

-  Python version 3.9 or higher
-  PyTorch (the latest version is recommended)
-  TorchVision (the latest version is recommended)
-  Captum (the latest version is recommended)
-  Matplotlib (the latest version is recommended)

To install Captum in a virtual environment, use:

.. code-block:: sh

    pip install torch torchvision captum matplotlib

Restart this notebook in the environment you set up, and you’re ready to
go!


A First Example
---------------

To start, let’s take a simple, visual example. We’ll start with a ResNet
model pretrained on the ImageNet dataset. We’ll get a test input, and
use different **Feature Attribution** algorithms to examine how the
input images affect the output, and see a helpful visualization of this
input attribution map for some test images.

First, some imports:

"""

# PyTorch 主包。后面会用它执行模型推理、取最大概率类别，并保存张量结果。
import torch
# functional 模块提供 softmax 等无状态函数。这里用 softmax 把模型输出转成概率。
import torch.nn.functional as F
# torchvision.transforms 提供常见图像预处理步骤，例如缩放、裁剪和转张量。
import torchvision.transforms as transforms
# torchvision.models 提供已经实现好的视觉模型。这里会直接加载预训练 ResNet-18。
import torchvision.models as models

# 导入 Captum 主包，保留原教程导入，便于读者知道当前示例依赖 Captum。
import captum
# IntegratedGradients 和 Occlusion 用于输入特征归因；
# LayerGradCam 和 LayerAttribution 用于解释模型中间层的响应。
from captum.attr import IntegratedGradients, Occlusion, LayerGradCam, LayerAttribution
# Captum 的 visualization 工具封装了常见的图像归因可视化方式。
from captum.attr import visualization as viz

# 保留原教程中的标准库导入。json 会用来读取 ImageNet 类别标签文件。
import os, sys
import json

# NumPy 用于在 PIL 图像、PyTorch 张量和 Matplotlib 显示格式之间转换。
import numpy as np
# PIL Image 用来读取本地示例图片。
from PIL import Image
# Matplotlib 用来显示原图和 Captum 生成的归因热力图。
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


#########################################################################
# Now we’ll use the TorchVision model library to download a pretrained
# ResNet. Since we’re not training, we’ll place it in evaluation mode for
# now.
#

# 加载在 ImageNet 上预训练过的 ResNet-18。
# weights='IMAGENET1K_V1' 表示使用 TorchVision 提供的一组 ImageNet 权重。
# 如果本地没有该权重文件，首次运行时 TorchVision 可能会下载它。
model = models.resnet18(weights='IMAGENET1K_V1')
# eval() 会把模型切换到推理模式。这样 BatchNorm、Dropout 等层会使用推理行为。
# 本教程只解释模型输出，不训练模型，所以需要使用 eval()。
model = model.eval()


#######################################################################
# The place where you got this interactive notebook should also have an
# ``img`` folder with a file ``cat.jpg`` in it.
#

# 读取示例猫图片。路径相对于运行 notebook 或脚本时的当前目录。
test_img = Image.open('img/cat.jpg')
# Matplotlib 更容易显示 NumPy 数组，所以先把 PIL Image 转成数组。
test_img_data = np.asarray(test_img)
# 显示原始图片，先让读者确认输入内容。
plt.imshow(test_img_data)
plt.show()


##########################################################################
# Our ResNet model was trained on the ImageNet dataset, and expects images
# to be of a certain size, with the channel data normalized to a specific
# range of values. We’ll also pull in the list of human-readable labels
# for the categories our model recognizes - that should be in the ``img``
# folder as well.
#

# model expects 224x224 3-color image
# ResNet-18 的 ImageNet 权重通常接收 224x224 的 RGB 图片。
# Compose 会把多个预处理步骤按顺序串起来。
transform = transforms.Compose([
    # Resize(224) 会把图片较短边缩放到 224，保持宽高比例。
    transforms.Resize(224),
    # CenterCrop(224) 从中心裁出 224x224 区域，确保输入尺寸固定。
    transforms.CenterCrop(224),
    # ToTensor() 把 PIL 图片转成 float 张量，形状从 [H, W, C] 变为 [C, H, W]，
    # 并把像素值缩放到 [0, 1]。
    transforms.ToTensor()
])

# standard ImageNet normalization
# ImageNet 预训练模型要求输入使用训练时相同的均值和标准差做归一化。
# mean 和 std 分别对应 RGB 三个通道。
transform_normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

# transformed_img 是显示友好的图像张量，形状是 [3, 224, 224]，数值范围约为 [0, 1]。
transformed_img = transform(test_img)
# input_img 是喂给模型的张量，已经按照 ImageNet 统计值归一化。
input_img = transform_normalize(transformed_img)
# 模型输入需要 batch 维度，单张图片也要变成 [1, 3, 224, 224]。
input_img = input_img.unsqueeze(0)  # the model requires a dummy batch dimension

# ImageNet 类别文件把类别编号映射到可读标签，例如 "tabby"、"tiger_cat"。
labels_path = 'img/imagenet_class_index.json'
with open(labels_path) as json_data:
    idx_to_labels = json.load(json_data)


######################################################################
# Now, we can ask the question: What does our model think this image
# represents?
#

# 对预处理后的图片做一次前向推理。ResNet 输出形状通常是 [1, 1000]，
# 其中 1000 对应 ImageNet 的 1000 个类别。
output = model(input_img)
# 模型原始输出是 logits，不是概率。softmax 会把它们转换成总和为 1 的概率分布。
# dim=1 表示沿类别维度做 softmax。
output = F.softmax(output, dim=1)
# topk(..., 1) 取概率最高的 1 个类别，返回概率值和类别索引。
prediction_score, pred_label_idx = torch.topk(output, 1)
# squeeze_() 是原地操作，把形状为 [1, 1] 或 [1] 的索引压成标量形状，方便后续使用。
pred_label_idx.squeeze_()
# JSON 文件的键是字符串形式的类别编号，所以这里把整数索引转成 str。
predicted_label = idx_to_labels[str(pred_label_idx.item())][1]
print('Predicted:', predicted_label, '(', prediction_score.squeeze().item(), ')')


######################################################################
# We’ve confirmed that ResNet thinks our image of a cat is, in fact, a
# cat. But *why* does the model think this is an image of a cat?
#
# For the answer to that, we turn to Captum.
#


##########################################################################
# Feature Attribution with Integrated Gradients
# ---------------------------------------------
#
# **Feature attribution** attributes a particular output to features of
# the input. It uses a specific input - here, our test image - to generate
# a map of the relative importance of each input feature to a particular
# output feature.
#
# `Integrated
# Gradients <https://captum.ai/api/integrated_gradients.html>`__ is one of
# the feature attribution algorithms available in Captum. Integrated
# Gradients assigns an importance score to each input feature by
# approximating the integral of the gradients of the model’s output with
# respect to the inputs.
#
# In our case, we’re going to be taking a specific element of the output
# vector - that is, the one indicating the model’s confidence in its
# chosen category - and use Integrated Gradients to understand what parts
# of the input image contributed to this output.
#
# Once we have the importance map from Integrated Gradients, we’ll use the
# visualization tools in Captum to give a helpful representation of the
# importance map. Captum’s ``visualize_image_attr()`` function provides a
# variety of options for customizing display of your attribution data.
# Here, we pass in a custom Matplotlib color map.
#
# Running the cell with the ``integrated_gradients.attribute()`` call will
# usually take a minute or two.
#

# Initialize the attribution algorithm with the model
# Integrated Gradients 属于基于梯度的特征归因方法。
# 它会比较输入图片与一个基线输入之间的路径，并沿路径累积梯度。
integrated_gradients = IntegratedGradients(model)

# Ask the algorithm to attribute our output target to
# target=pred_label_idx 表示解释“模型预测出的那个类别”的得分。
# n_steps=200 表示从基线到真实输入之间取 200 个积分近似点；
# 步数越大通常越精细，但计算时间也越长。
attributions_ig = integrated_gradients.attribute(input_img, target=pred_label_idx, n_steps=200)

# Show the original image for comparison
# visualize_image_attr 的第一个参数是归因数据。显示原图时不需要归因数据，所以传 None。
# np.transpose(..., (1, 2, 0)) 把 PyTorch 的 [C, H, W] 转为 Matplotlib 的 [H, W, C]。
_ = viz.visualize_image_attr(None, np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                             method="original_image", title="Original Image")

# 自定义一个白到蓝的色图。白色表示归因弱，蓝色表示正向归因更强。
default_cmap = LinearSegmentedColormap.from_list('custom blue',
                                                 [(0, '#ffffff'),
                                                  (0.25, '#0000ff'),
                                                  (1, '#0000ff')], N=256)

# attributions_ig 的形状与 input_img 类似，是 [1, 3, 224, 224]。
# squeeze() 去掉 batch 维度，detach() 从计算图分离，cpu().numpy() 转成可视化需要的数组。
# sign='positive' 只显示支持目标类别的正向贡献区域。
_ = viz.visualize_image_attr(np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                             np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                             method='heat_map',
                             cmap=default_cmap,
                             show_colorbar=True,
                             sign='positive',
                             title='Integrated Gradients')


#######################################################################
# In the image above, you should see that Integrated Gradients gives us
# the strongest signal around the cat’s location in the image.
#


##########################################################################
# Feature Attribution with Occlusion
# ----------------------------------
#
# Gradient-based attribution methods help to understand the model in terms
# of directly computing out the output changes with respect to the input.
# *Perturbation-based attribution* methods approach this more directly, by
# introducing changes to the input to measure the effect on the output.
# `Occlusion <https://captum.ai/api/occlusion.html>`__ is one such method.
# It involves replacing sections of the input image, and examining the
# effect on the output signal.
#
# Below, we set up Occlusion attribution. Similarly to configuring a
# convolutional neural network, you can specify the size of the target
# region, and a stride length to determine the spacing of individual
# measurements. We’ll visualize the output of our Occlusion attribution
# with ``visualize_image_attr_multiple()``, showing heat maps of both
# positive and negative attribution by region, and by masking the original
# image with the positive attribution regions. The masking gives a very
# instructive view of what regions of our cat photo the model found to be
# most “cat-like”.
#

# Occlusion 属于基于扰动的归因方法。
# 它会用一个基线值遮挡图片的小区域，再观察目标类别分数如何变化。
occlusion = Occlusion(model)

# attribute() 会按滑动窗口不断遮挡输入图像，并统计每个区域对预测结果的影响。
attributions_occ = occlusion.attribute(input_img,
                                       target=pred_label_idx,
                                       # strides 对应 [通道, 高, 宽] 三个维度的步长。
                                       # 通道步长为 3 表示一次覆盖 RGB 三个通道；
                                       # 高和宽步长为 8 表示遮挡窗口每次移动 8 个像素。
                                       strides=(3, 8, 8),
                                       # sliding_window_shapes 是每次遮挡的窗口大小。
                                       # (3, 15, 15) 表示同时遮挡 3 个颜色通道中的 15x15 像素块。
                                       sliding_window_shapes=(3, 15, 15),
                                       # baselines=0 表示被遮挡区域用 0 替换。
                                       # 因为 input_img 已经归一化，0 大致表示接近平均水平的输入值。
                                       baselines=0)


# visualize_image_attr_multiple 可以一次画多张解释图，便于横向比较。
# 这里依次显示：原图、正向归因热力图、负向归因热力图、正向归因遮罩图。
_ = viz.visualize_image_attr_multiple(np.transpose(attributions_occ.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                      np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                      ["original_image", "heat_map", "heat_map", "masked_image"],
                                      ["all", "positive", "negative", "positive"],
                                      show_colorbar=True,
                                      titles=["Original", "Positive Attribution", "Negative Attribution", "Masked"],
                                      fig_size=(18, 6)
                                      )


######################################################################
# Again, we see greater significance placed on the region of the image
# that contains the cat.
#


#########################################################################
# Layer Attribution with Layer GradCAM
# ------------------------------------
#
# **Layer Attribution** allows you to attribute the activity of hidden
# layers within your model to features of your input. Below, we’ll use a
# layer attribution algorithm to examine the activity of one of the
# convolutional layers within our model.
#
# GradCAM computes the gradients of the target output with respect to the
# given layer, averages for each output channel (dimension 2 of output),
# and multiplies the average gradient for each channel by the layer
# activations. The results are summed over all channels. GradCAM is
# designed for convnets; since the activity of convolutional layers often
# maps spatially to the input, GradCAM attributions are often upsampled
# and used to mask the input.
#
# Layer attribution is set up similarly to input attribution, except that
# in addition to the model, you must specify a hidden layer within the
# model that you wish to examine. As above, when we call ``attribute()``,
# we specify the target class of interest.
#

# LayerGradCam 解释的是某个中间层的空间响应，而不是直接解释输入像素。
# model.layer3[1].conv2 是 ResNet-18 中第 3 组残差层里的一个卷积层。
# 选择较深的卷积层时，热力图通常更接近语义区域，例如猫的身体或脸部。
layer_gradcam = LayerGradCam(model, model.layer3[1].conv2)
# 同样指定 target=pred_label_idx，表示观察这个卷积层如何支持当前预测类别。
attributions_lgc = layer_gradcam.attribute(input_img, target=pred_label_idx)

# Layer GradCAM 的输出空间尺寸通常小于原图，因为中间卷积层经过了下采样。
# permute(1, 2, 0) 把通道维放到最后，方便 Captum 的图像可视化函数处理。
_ = viz.visualize_image_attr(attributions_lgc[0].cpu().permute(1, 2, 0).detach().numpy(),
                             sign="all",
                             title="Layer 3 Block 1 Conv 2")


##########################################################################
# We’ll use the convenience method ``interpolate()`` in the
# `LayerAttribution <https://captum.ai/api/base_classes.html?highlight=layerattribution#captum.attr.LayerAttribution>`__
# base class to upsample this attribution data for comparison to the input
# image.
#

# interpolate() 会把较小的层归因图上采样到输入图片大小。
# input_img.shape[2:] 取的是输入图片的高和宽，也就是 (224, 224)。
upsamp_attr_lgc = LayerAttribution.interpolate(attributions_lgc, input_img.shape[2:])

# 打印三个张量形状，帮助读者理解“层归因图”和“输入图像”尺寸之间的关系。
# attributions_lgc 通常形状较小；upsamp_attr_lgc 会被放大到和 input_img 的高宽一致。
print(attributions_lgc.shape)
print(upsamp_attr_lgc.shape)
print(input_img.shape)

# 把上采样后的 Layer GradCAM 结果叠加到原图上。
# blended_heat_map 可以看到热力图和原图的对应关系；
# masked_image 只突出正向贡献较强的区域。
_ = viz.visualize_image_attr_multiple(upsamp_attr_lgc[0].cpu().permute(1, 2, 0).detach().numpy(),
                                      transformed_img.permute(1, 2, 0).numpy(),
                                      ["original_image", "blended_heat_map", "masked_image"],
                                      ["all", "positive", "positive"],
                                      show_colorbar=True,
                                      titles=["Original", "Positive Attribution", "Masked"],
                                      fig_size=(18, 6))


#######################################################################
# Visualizations such as this can give you novel insights into how your
# hidden layers respond to your input.
#
