# 如何使用3D模型组件

相关空间：https://huggingface.co/spaces/dawood/Model3D, https://huggingface.co/spaces/radames/PIFu-Clothed-Human-Digitization, https://huggingface.co/spaces/radames/dpt-depth-estimation-3d-obj
标签：VISION, IMAGE

## 介绍

机器学习中的3D模型越来越受欢迎，并且是一些最有趣的演示实验。使用`gradio`，您可以轻松构建您的3D图像模型的演示，并与任何人分享。Gradio 3D模型组件接受3种文件类型，包括：*.obj*，*.glb*和*.gltf*。

本指南将向您展示如何使用几行代码构建您的3D图像模型的演示；像下面这个示例一样。点击、拖拽和缩放来玩转3D对象：

<gradio-app space="dawood/Model3D"> </gradio-app>

### 先决条件

确保已经[安装](https://gradio.app/quickstart)了`gradio` Python包。


## 查看代码

让我们来看看如何创建上面的最简界面。在这种情况下，预测函数将只返回原始的3D模型网格，但您可以更改此函数以在您的机器学习模型上运行推理。我们将在下面看更复杂的示例。

```python
import gradio as gr

def load_mesh(mesh_file_name):
    return mesh_file_name

demo = gr.Interface(
    fn=load_mesh,
    inputs=gr.Model3D(),
    outputs=gr.Model3D(clear_color=[0.0, 0.0, 0.0, 0.0],  label="3D Model"),
    examples=[
        ["files/Bunny.obj"],
        ["files/Duck.glb"],
        ["files/Fox.gltf"],
        ["files/face.obj"],
    ],
    cache_examples=True,
)

demo.launch()
```

让我们来解析上面的代码：

`load_mesh`：这是我们的“预测”函数，为简单起见，该函数将接收3D模型网格并返回它。

创建界面：

* `fn`：当用户点击提交时使用的预测函数。在我们的例子中，它是`load_mesh`函数。
* `inputs`：创建一个model3D输入组件。输入是一个上传的文件，作为{str}文件路径。
* `outputs`：创建一个model3D输出组件。输出组件也期望一个文件作为{str}文件路径。
  * `clear_color`：这是3D模型画布的背景颜色。期望RGBa值。
  * `label`：出现在组件左上角的标签。
* `examples`：3D模型文件的列表。3D模型组件可以接受*.obj*，*.glb*和*.gltf*文件类型。
* `cache_examples`：保存示例的预测输出，以节省推理时间。


## 探索更复杂的Model3D演示

下面是一个使用DPT模型预测图像深度，然后使用3D点云创建3D对象的演示。查看[code.py](https://huggingface.co/spaces/radames/dpt-depth-estimation-3d-obj/blob/main/app.py)文件，了解代码和模型预测函数。
<gradio-app space="radames/dpt-depth-estimation-3d-obj"> </gradio-app>

下面是一个使用PIFu模型将穿着衣物的人的图像转换为3D数字化模型的演示。查看[spaces.py](https://huggingface.co/spaces/radames/PIFu-Clothed-Human-Digitization/blob/main/PIFu/spaces.py)文件，了解代码和模型预测函数。

<gradio-app space="radames/PIFu-Clothed-Human-Digitization"> </gradio-app>

----------

搞定！这就是构建Model3D模型界面所需的所有代码。以下是一些您可能会发现有用的参考资料：

* Gradio的[“入门指南”](https://gradio.app/getting_started/)
* 第一个[3D模型演示](https://huggingface.co/spaces/dawood/Model3D)和[完整代码](https://huggingface.co/spaces/dawood/Model3D/tree/main)（在Hugging Face Spaces上）
