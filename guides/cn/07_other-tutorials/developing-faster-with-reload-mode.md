# 通过自动重载实现更快的开发

**先决条件**：本指南要求您了解块的知识。请确保[先阅读块指南](https://gradio.app/quickstart/#blocks-more-flexibility-and-control)。

本指南介绍了自动重新加载、在Python IDE中重新加载以及在Jupyter Notebooks中使用gradio的方法。

## 为什么要使用自动重载？

当您构建Gradio演示时，特别是使用Blocks构建时，您可能会发现反复运行代码以测试更改很麻烦。

为了更快速、更便捷地编写代码，我们已经简化了在**Python IDE**（如VS Code、Sublime Text、PyCharm等）中开发或从终端运行Python代码时“重新加载”Gradio应用的方式。我们还开发了一个类似的“魔法命令”，使您可以更快速地重新运行单元格，如果您使用Jupyter Notebooks（或类似的环境，如Colab）的话。

这个简短的指南将涵盖这两种方法，所以无论您如何编写Python代码，您都将知道如何更快地构建Gradio应用程序。

## Python IDE 重载 🔥

如果您使用Python IDE构建Gradio Blocks，那么代码文件（假设命名为`run.py`）可能如下所示：

```python
import gradio as gr

with gr.Blocks() as demo:
    gr.Markdown("# 来自Gradio的问候！")
    inp = gr.Textbox(placeholder="您叫什么名字？")
    out = gr.Textbox()

    inp.change(fn=lambda x: f"欢迎，{x}！",
               inputs=inp,
               outputs=out)

if __name__ == "__main__":
    demo.launch()
```

问题在于，每当您想要更改布局、事件或组件时，都必须通过编写`python run.py`来关闭和重新运行应用程序。

而不是这样做，您可以通过更改1个单词来以**重新加载模式**运行代码：将 `python` 更改为 `gradio`：

在终端中运行 `gradio run.py`。就是这样！

现在，您将看到类似于这样的内容：

```bash
Launching in *reload mode* on: http://127.0.0.1:7860 (Press CTRL+C to quit)

Watching...

WARNING:  The --reload flag should not be used in production on Windows.
```

这里最重要的一行是 `正在观察...`。这里发生的情况是Gradio将观察`run.py`文件所在的目录，如果文件发生更改，它将自动为您重新运行文件。因此，您只需专注于编写代码，Gradio演示将自动刷新 🥳

⚠️ 警告：`gradio`命令不会检测传递给`launch()`方法的参数，因为在重新加载模式下从未调用`launch()`方法。例如，设置`launch()`中的`auth`或`show_error`不会在应用程序中反映出来。

当您使用重新加载模式时，请记住一件重要的事情：Gradio专门查找名为 `demo` 的Gradio Blocks/Interface演示。如果您将演示命名为其他名称，您需要在代码中的第二个参数中传入演示的FastAPI应用程序的名称。对于Gradio演示，可以使用`.app`属性访问FastAPI应用程序。因此，如果您的 `run.py` 文件如下所示：

```python
import gradio as gr

with gr.Blocks() as my_demo:
    gr.Markdown("# 来自Gradio的问候！")
    inp = gr.Textbox(placeholder="您叫什么名字？")
    out = gr.Textbox()

    inp.change(fn=lambda x: f"欢迎，{x}！",
               inputs=inp,
               outputs=out)

if __name__ == "__main__":
    my_demo.launch()
```

那么您可以这样启动它：`gradio run.py my_demo.app`。

🔥 如果您的应用程序接受命令行参数，您也可以传递它们。下面是一个例子：

```python
import gradio as gr
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="User")
args, unknown = parser.parse_known_args()

with gr.Blocks() as demo:
    gr.Markdown(f"# 欢迎 {args.name}！")
    inp = gr.Textbox()
    out = gr.Textbox()

    inp.change(fn=lambda x: x, inputs=inp, outputs=out)

if __name__ == "__main__":
    demo.launch()
```

您可以像这样运行它：`gradio run.py --name Gretel`

作为一个小提示，只要更改了`run.py`源代码或Gradio源代码，自动重新加载就会发生。这意味着如果您决定[为Gradio做贡献](https://github.com/gradio-app/gradio/blob/main/CONTRIBUTING.md)，这将非常有用 ✅

## Jupyter Notebook魔法命令🔮

如果您使用Jupyter Notebooks（或Colab Notebooks等）进行开发，我们也为您提供了一个解决方案！

我们开发了一个**magic command 魔法命令**，可以为您创建和运行一个Blocks演示。要使用此功能，在笔记本顶部加载gradio扩展：

`%load_ext gradio`

然后，在您正在开发Gradio演示的单元格中，只需在顶部写入魔法命令**`%%blocks`**，然后像平常一样编写布局和组件：

```py
%%blocks

import gradio as gr

gr.Markdown("# 来自Gradio的问候！")
inp = gr.Textbox(placeholder="您叫什么名字？")
out = gr.Textbox()

inp.change(fn=lambda x: f"欢迎，{x}！",
           inputs=inp,
           outputs=out)
```

请注意：

* 您不需要放置样板代码 `with gr.Blocks() as demo:` 和 `demo.launch()` — Gradio会自动为您完成！

* 每次重新运行单元格时，Gradio都将在相同的端口上重新启动您的应用程序，并使用相同的底层网络服务器。这意味着您将比正常重新运行单元格更快地看到变化。

下面是在Jupyter Notebook中的示例：

![](https://i.ibb.co/nrszFws/Blocks.gif)

🪄这在colab笔记本中也适用！[这是一个colab笔记本](https://colab.research.google.com/drive/1jUlX1w7JqckRHVE-nbDyMPyZ7fYD8488?authuser=1#scrollTo=zxHYjbCTTz_5)，您可以在其中看到Blocks魔法效果。尝试进行一些更改并重新运行带有Gradio代码的单元格！

Notebook Magic现在是作者构建Gradio演示的首选方式。无论您如何编写Python代码，我们都希望这两种方法都能为您提供更好的Gradio开发体验。

--------

## 下一步

既然您已经了解了如何使用Gradio快速开发，请开始构建自己的应用程序吧！

如果你正在寻找灵感，请尝试浏览其他人用Gradio构建的演示，[浏览Hugging Face Spaces](http://hf.space/) 🤗

