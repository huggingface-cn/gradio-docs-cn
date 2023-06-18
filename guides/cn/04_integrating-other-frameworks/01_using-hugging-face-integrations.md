# 使用Hugging Face集成

相关空间：https://huggingface.co/spaces/gradio/helsinki_translation_en_es
标签：HUB，SPACES，EMBED

由<a href="https://huggingface.co/osanseviero">Omar Sanseviero</a>贡献🦙

## 介绍

Hugging Face Hub是一个集成平台，拥有超过190,000个[模型](https://huggingface.co/models)，32,000个[数据集](https://huggingface.co/datasets)和40,000个[演示](https://huggingface.co/spaces)，也被称为Spaces。虽然Hugging Face以其🤗 transformers和diffusers库而闻名，但Hub还支持许多机器学习库，如PyTorch，TensorFlow，spaCy等，涵盖了从计算机视觉到强化学习等各个领域。

Gradio拥有多个功能，使其非常容易利用Hub上的现有模型和Spaces。本指南将介绍这些功能。

## 使用`pipeline`进行常规推理

首先，让我们构建一个简单的界面，将英文翻译成西班牙文。在赫尔辛基大学共享的一千多个模型中，有一个[现有模型](https://huggingface.co/Helsinki-NLP/opus-mt-en-es)，名为`opus-mt-en-es`，可以正好做到这一点！

🤗 transformers库有一个非常易于使用的抽象层，[`pipeline()`](https://huggingface.co/docs/transformers/v4.16.2/en/main_classes/pipelines#transformers.pipeline)处理大部分复杂代码，为常见任务提供简单的API。通过指定任务和（可选）模型，您可以使用几行代码使用现有模型：

```python
import gradio as gr

from transformers import pipeline

pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-en-es")

def predict(text):
  return pipe(text)[0]["translation_text"]
  
demo = gr.Interface(
  fn=predict, 
  inputs='text',
  outputs='text',
)

demo.launch()
```

但是，`gradio`实际上使将`pipeline`转换为演示更加容易，只需使用`gradio.Interface.from_pipeline`方法，无需指定输入和输出组件：

```python
from transformers import pipeline
import gradio as gr

pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-en-es")

demo = gr.Interface.from_pipeline(pipe)
demo.launch()
```

上述代码生成了以下界面，您可以在浏览器中直接尝试：


<gradio-app space="Helsinki-NLP/opus-mt-en-es"></gradio-app>


## 使用Hugging Face推理API

Hugging Face提供了一个名为[Inference API](https://huggingface.co/inference-api)的免费服务，允许您向Hub中的模型发送HTTP请求。对于基于transformers或diffusers的模型，API的速度可以比自己运行推理快2到10倍。该API是免费的（受速率限制），您可以在想要在生产中使用时切换到专用的[推理端点](https://huggingface.co/pricing)。

让我们尝试使用推理API而不是自己加载模型的方式进行相同的演示。鉴于Inference API支持的Hugging Face模型，Gradio可以自动推断出预期的输入和输出，并进行底层服务器调用，因此您不必担心定义预测函数。以下是代码示例！

```python
import gradio as gr

demo = gr.load("Helsinki-NLP/opus-mt-en-es", src="models")

demo.launch()
```

请注意，我们只需指定模型名称并说明`src`应为`models`（Hugging Face的Model Hub）。由于您不会在计算机上加载模型，因此无需安装任何依赖项（除了`gradio`）。

您可能会注意到，第一次推理大约需要20秒。这是因为推理API正在服务器中加载模型。之后您会获得一些好处：

* 推理速度更快。
* 服务器缓存您的请求。
* 您获得内置的自动缩放功能。

## 托管您的Gradio演示

[Hugging Face Spaces](https://hf.co/spaces)允许任何人免费托管其Gradio演示，上传Gradio演示只需几分钟。您可以前往[hf.co/new-space](https://huggingface.co/new-space)，选择Gradio SDK，创建一个`app.py`文件，完成！您将拥有一个可以与任何人共享的演示。要了解更多信息，请阅读[此指南以使用网站在Hugging Face Spaces上托管](https://huggingface.co/blog/gradio-spaces)。


或者，您可以通过使用[huggingface_hub client library](https://huggingface.co/docs/huggingface_hub/index)库来以编程方式创建一个Space。这是一个示例：

```python
from huggingface_hub import (
    create_repo,
    get_full_repo_name,
    upload_file,
)
create_repo(name=target_space_name, token=hf_token, repo_type="space", space_sdk="gradio")
repo_name = get_full_repo_name(model_id=target_space_name, token=hf_token)
file_url = upload_file(
    path_or_fileobj="file.txt",
    path_in_repo="app.py",
    repo_id=repo_name,
    repo_type="space",
    token=hf_token,
)
```
在这里，`create_repo`使用特定帐户的Write Token在特定帐户下创建一个带有目标名称的gradio repo。`repo_name`获取相关存储库的完整存储库名称。最后，`upload_file`将文件上传到存储库中，并将其命名为`app.py`。



## 在其他网站上嵌入您的Space演示

在本指南中，您已经看到了许多嵌入的Gradio演示。您也可以在自己的网站上这样做！第一步是创建一个包含您想展示的演示的Hugging Face Space。然后，[按照此处的步骤将Space嵌入到您的网站上](/sharing-your-app/#embedding-hosted-spaces)。


## 从Spaces加载演示

您还可以在Hugging Face Spaces上使用和混合现有的Gradio演示。例如，您可以将两个现有的Gradio演示放在单独的选项卡中并创建一个新的演示。您可以在本地运行此新演示，或将其上传到Spaces，为混合和创建新的演示提供无限可能性！

以下是一个完全实现此目标的示例：

```python
import gradio as gr

with gr.Blocks() as demo:
  with gr.Tab("Translate to Spanish"):
    gr.load("gradio/helsinki_translation_en_es", src="spaces")
  with gr.Tab("Translate to French"):
    gr.load("abidlabs/en2fr", src="spaces")

demo.launch()
```

请注意，我们使用了`gr.load()`，这与使用推理API加载模型所使用的方法相同。但是，在这里，我们指定`src`为`spaces`（Hugging Face Spaces）。

## 小结

就是这样！让我们回顾一下Gradio和Hugging Face共同工作的各种方式：

1. 您可以使用`from_pipeline()`将`transformers` pipeline转换为Gradio演示
2. 您可以使用`gr.load()`轻松地围绕推理API构建演示，而无需加载模型
3. 您可以在Hugging Face Spaces上托管您的Gradio演示，可以使用GUI或完全使用Python。
4. 您可以将托管在Hugging Face Spaces上的Gradio演示嵌入到自己的网站上。
5. 您可以使用`gr.load()`从Hugging Face Spaces加载演示，以重新混合和创建新的Gradio演示。


🤗
