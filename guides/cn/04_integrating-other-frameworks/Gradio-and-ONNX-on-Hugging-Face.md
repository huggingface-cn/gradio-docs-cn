# Gradio和ONNX在Hugging Face上

Related spaces: https://huggingface.co/spaces/onnx/EfficientNet-Lite4
Tags: ONNX，SPACES
由Gradio和<a href="https://onnx.ai/">ONNX</a>团队贡献

## 介绍

在这个指南中，我们将为您介绍以下内容：

* ONNX、ONNX模型仓库、Gradio和Hugging Face Spaces的介绍
* 如何为EfficientNet-Lite4设置Gradio演示
* 如何为Hugging Face上的ONNX组织贡献自己的Gradio演示

下面是一个ONNX模型的示例：在下面尝试EfficientNet-Lite4演示。

<iframe src="https://onnx-efficientnet-lite4.hf.space" frameBorder="0" height="810" title="Gradio app" class="container p-0 flex-grow space-iframe" allow="accelerometer; ambient-light-sensor; autoplay; battery; camera; document-domain; encrypted-media; fullscreen; geolocation; gyroscope; layout-animations; legacy-image-formats; magnetometer; microphone; midi; oversized-images; payment; picture-in-picture; publickey-credentials-get; sync-xhr; usb; vr ; wake-lock; xr-spatial-tracking" sandbox="allow-forms allow-modals allow-popups allow-popups-to-escape-sandbox allow-same-origin allow-scripts allow-downloads"></iframe>

## ONNX模型仓库是什么？
Open Neural Network Exchange（[ONNX](https://onnx.ai/)）是一种表示机器学习模型的开放标准格式。ONNX由一个实现了该格式的合作伙伴社区支持，该社区将其实施到许多框架和工具中。例如，如果您在TensorFlow或PyTorch中训练了一个模型，您可以轻松地将其转换为ONNX，然后使用类似ONNX Runtime的引擎/编译器在各种设备上运行它。

[ONNX模型仓库](https://github.com/onnx/models)是由社区成员贡献的一组预训练的先进模型，格式为ONNX。每个模型都附带了用于模型训练和运行推理的Jupyter笔记本。这些笔记本以Python编写，并包含到训练数据集的链接，以及描述模型架构的原始论文的参考文献。


## Hugging Face Spaces和Gradio是什么？

### Gradio

Gradio可让用户使用Python代码将其机器学习模型演示为Web应用程序。Gradio将Python函数封装到用户界面中，演示可以在jupyter笔记本、colab笔记本中启动，并可以嵌入到您自己的网站上，并在Hugging Face Spaces上免费托管。

在此处开始[https://gradio.app/getting_started](https://gradio.app/getting_started)

### Hugging Face Spaces

Hugging Face Spaces是Gradio演示的免费托管选项。Spaces提供了3种SDK选项：Gradio、Streamlit和静态HTML演示。Spaces可以是公共的或私有的，工作流程与github repos类似。目前Hugging Face上有2000多个Spaces。在此处了解更多关于Spaces的信息[https://huggingface.co/spaces/launch](https://huggingface.co/spaces/launch)。

### Hugging Face模型

Hugging Face模型中心还支持ONNX模型，并且可以通过[ONNX标签](https://huggingface.co/models?library=onnx&sort=downloads)对ONNX模型进行筛选

## Hugging Face是如何帮助ONNX模型仓库的？
ONNX模型仓库中有许多Jupyter笔记本供用户测试模型。以前，用户需要自己下载模型并在本地运行这些笔记本测试。有了Hugging Face，测试过程可以更简单和用户友好。用户可以在Hugging Face Spaces上轻松尝试ONNX模型仓库中的某个模型，并使用ONNX Runtime运行由Gradio提供支持的快速演示，全部在云端进行，无需在本地下载任何内容。请注意，ONNX有各种运行时，例如[ONNX Runtime](https://github.com/microsoft/onnxruntime)、[MXNet](https://github.com/apache/incubator-mxnet)等

## ONNX Runtime的作用是什么？
ONNX Runtime是一个跨平台的推理和训练机器学习加速器。它使得在Hugging Face上使用ONNX模型仓库中的模型进行实时Gradio演示成为可能。

ONNX Runtime可以实现更快的客户体验和更低的成本，支持来自PyTorch和TensorFlow/Keras等深度学习框架以及scikit-learn、LightGBM、XGBoost等传统机器学习库的模型。ONNX Runtime与不同的硬件、驱动程序和操作系统兼容，并通过利用适用的硬件加速器以及图形优化和转换提供最佳性能。有关更多信息，请参阅[官方网站](https://onnxruntime.ai/)。

## 为EfficientNet-Lite4设置Gradio演示

EfficientNet-Lite 4是EfficientNet-Lite系列中最大和最准确的模型。它是一个仅使用整数量化的模型，能够在所有EfficientNet模型中提供最高的准确率。在Pixel 4 CPU上以实时方式运行（例如30ms/图像）时，可以实现80.4％的ImageNet top-1准确率。要了解更多信息，请阅读[模型卡片](https://github.com/onnx/models/tree/main/vision/classification/efficientnet-lite4)

在这里，我们将演示如何使用Gradio为EfficientNet-Lite4设置示例演示

首先，我们导入所需的依赖项并下载和载入来自ONNX模型仓库的efficientnet-lite4模型。然后从labels_map.txt文件加载标签。接下来，我们设置预处理函数、加载用于推理的模型并设置推理函数。最后，将推理函数封装到Gradio接口中，供用户进行交互。下面是完整的代码。


```python
import numpy as np
import math
import matplotlib.pyplot as plt
import cv2
import json
import gradio as gr
from huggingface_hub import hf_hub_download
from onnx import hub
import onnxruntime as ort

# 从ONNX模型仓库加载ONNX模型
model = hub.load("efficientnet-lite4")
# 加载标签文本文件
labels = json.load(open("labels_map.txt", "r"))

# 通过将图像从中心调整大小并裁剪到224x224来设置图像文件的尺寸
def pre_process_edgetpu(img, dims):
    output_height, output_width, _ = dims
    img = resize_with_aspectratio(img, output_height, output_width, inter_pol=cv2.INTER_LINEAR)
    img = center_crop(img, output_height, output_width)
    img = np.asarray(img, dtype='float32')
    # 将jpg像素值从[0 - 255]转换为浮点数组[-1.0 - 1.0]
    img -= [127.0, 127.0, 127.0]
    img /= [128.0, 128.0, 128.0]
    return img

# 使用等比例缩放调整图像尺寸
def resize_with_aspectratio(img, out_height, out_width, scale=87.5, inter_pol=cv2.INTER_LINEAR):
    height, width, _ = img.shape
    new_height = int(100. * out_height / scale)
    new_width = int(100. * out_width / scale)
    if height > width:
        w = new_width
        h = int(new_height * height / width)
    else:
        h = new_height
        w = int(new_width * width / height)
    img = cv2.resize(img, (w, h), interpolation=inter_pol)
    return img

# crops the image around the center based on given height and width
def center_crop(img, out_height, out_width):
    height, width, _ = img.shape
    left = int((width - out_width) / 2)
    right = int((width + out_width) / 2)
    top = int((height - out_height) / 2)
    bottom = int((height + out_height) / 2)
    img = img[top:bottom, left:right]
    return img


sess = ort.InferenceSession(model)

def inference(img):
  img = cv2.imread(img)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  
  img = pre_process_edgetpu(img, (224, 224, 3))
  
  img_batch = np.expand_dims(img, axis=0)

  results = sess.run(["Softmax:0"], {"images:0": img_batch})[0]
  result = reversed(results[0].argsort()[-5:])
  resultdic = {}
  for r in result:
      resultdic[labels[str(r)]] = float(results[0][r])
  return resultdic
  
title = "EfficientNet-Lite4"
description = "EfficientNet-Lite 4是最大的变体，也是EfficientNet-Lite模型集合中最准确的。它是一个仅包含整数的量化模型，具有所有EfficientNet模型中最高的准确度。在Pixel 4 CPU上，它实现了80.4％的ImageNet top-1准确度，同时仍然可以实时运行（例如30ms/图像）。"
examples = [['catonnx.jpg']]
gr.Interface(inference, gr.Image(type="filepath"), "label", title=title, description=description, examples=examples).launch()
```


## 如何使用ONNX模型在HF Spaces上贡献Gradio演示

* 将模型添加到[onnx model zoo](https://github.com/onnx/models/blob/main/.github/PULL_REQUEST_TEMPLATE.md)
* 在Hugging Face上创建一个账号[here](https://huggingface.co/join).
* 要查看还有哪些模型需要添加到ONNX组织中，请参阅[Models list](https://github.com/onnx/models#models)中的列表
* 在您的用户名下添加Gradio Demo，请参阅此[博文](https://huggingface.co/blog/gradio-spaces)以在Hugging Face上设置Gradio Demo。
* 请求加入ONNX组织[here](https://huggingface.co/onnx).
* 一旦获准，将模型从您的用户名下转移到ONNX组织
* 在模型表中为模型添加徽章，在[Models list](https://github.com/onnx/models#models)中查看示例
