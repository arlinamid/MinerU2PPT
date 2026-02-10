[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/arlinamid/MinerU2PPT)](https://github.com/arlinamid/MinerU2PPT/releases)
[![Downloads](https://img.shields.io/github/downloads/arlinamid/MinerU2PPT/total)](https://github.com/arlinamid/MinerU2PPT/releases)
[![Stars](https://img.shields.io/github/stars/arlinamid/MinerU2PPT)](https://github.com/arlinamid/MinerU2PPT/stargazers)
[![Issues](https://img.shields.io/github/issues/arlinamid/MinerU2PPT)](https://github.com/arlinamid/MinerU2PPT/issues)

# MinerU 转 PPTX 转换器

本工具利用 [MinerU PDF 提取器](https://mineru.net/OpenSourceTools/Extractor) 生成的结构化数据，将 PDF 文件和图片文件转换为可编辑的 PowerPoint 演示文稿（`.pptx`）。它能准确地重建文本、图片和布局，提供一个高保真的、可编辑的原始文档副本。

> **🙏 致谢**: 本项目受到 [JuniverseCoder 的 MinerU2PPT](https://github.com/JuniverseCoder/MinerU2PPT) 启发并在此基础上进行扩展。我们添加了重要的增强功能，包括AI驱动的文本校正、多语言支持、高级渲染功能和全面的开发工具。

本应用带有一个为用户设计的图形界面（GUI），简单易用。

![GUI Screenshot](img/gui.png)

## 🚀 快速开始

### 第一步：获取 MinerU JSON 数据
选择您喜欢的方式提取文档结构：

**🌐 网页版（推荐）**
- 访问：[MinerU PDF 提取器](https://mineru.net/OpenSourceTools/Extractor)
- 上传PDF/图像 → 下载JSON

**💻 桌面应用** 
- 下载：[MinerU 桌面应用](https://webpub.shlab.tech/MinerU/latest/win/MinerU-0.12.0-setup.exe?t=1)
- 安装 → 处理文件 → 导出JSON

### 第二步：下载 MinerU2PPTX
- 获取最新的 [MinerU2PPTX.exe](https://github.com/arlinamid/MinerU2PPT/releases)（无需安装）

### 第三步：转换
- 运行 MinerU2PPTX.exe → 选择原始文件 + JSON → 开始转换！

---

## 用户指南：如何使用

作为普通用户，您只需要独立的 `MinerU2PPT.exe` 文件。您无需安装 Python 或任何编程库。

1.  **下载应用程序**: 从本项目的 [Releases 页面](https://github.com/YOUR_USERNAME/YOUR_REPO/releases) 下载最新的 `.exe` 可执行文件。

2.  **获取 MinerU JSON 文件**:
    
    您有两种方式来提取转换所需的JSON数据：
    
    **方式A: 网页版提取器（推荐）**
    -   访问 [MinerU PDF/图片提取器](https://mineru.net/OpenSourceTools/Extractor) 网站
    -   上传您的PDF或图片文件，等待其处理完成
    -   下载生成的JSON文件
    
    **方式B: 桌面应用程序**
    -   下载 [MinerU 桌面应用](https://webpub.shlab.tech/MinerU/latest/win/MinerU-0.12.0-setup.exe?t=1)（Windows版）
    -   本地安装并运行应用程序
    -   处理您的PDF/图片文件以生成JSON数据
    
    JSON文件包含了本工具进行准确转换所需的文档结构信息。
    ![下载 JSON](img/download_json.png)

3.  **运行转换器**:
    -   双击 `.exe` 文件以启动应用程序。
    -   **选择输入文件**: 将您的 PDF 或图片文件拖拽到第一个输入框中，或使用“浏览...”按钮选择。
    -   **选择 JSON 文件**: 将您从 MinerU 下载的 JSON 文件拖拽到第二个输入框中。
    -   **输出路径**: 您的新 PowerPoint 文件的输出路径将被自动填充。您也可以直接输入路径，或使用“另存为...”按钮来更改。
    -   **选项**:
        -   **移除水印**: 勾选此项可自动擦除 MinerU JSON 中标记为“丢弃”的元素，例如页脚或页码。
        -   **生成调试图片**: 除非您需要排查问题，否则请勿勾选此项。
    -   点击 **开始转换**。

4.  **打开您的文件**: 转换完成后，点击“打开输出文件夹”按钮，即可找到您新生成的 `.pptx` 文件。

### 使用批量模式

该应用还支持批量模式，可以一次性转换多个文件。

1.  **切换到批量模式**: 点击应用右上角的“批量模式”按钮。界面将切换到批量处理视图。
2.  **添加任务**:
    -   点击“添加任务”按钮，会弹出一个新窗口。
    -   在弹出窗口中，通过拖拽或浏览选择**输入文件**、对应的 **MinerU JSON 文件**，并指定**输出路径**。
    -   为此特定任务设置**移除水印**选项。
    -   点击“确定”将任务添加到列表中。
3.  **管理任务**: 您可以向列表中添加多个任务。如果需要删除任务，请从列表中选中它，然后点击“删除任务”。
4.  **开始批量转换**: 添加完所有任务后，点击“开始批量转换”。应用将按顺序处理每个任务。日志区域会显示每个文件的处理进度。

## 开发者指南

本部分为需要从源代码运行或打包分发本应用的开发者提供说明。

### 环境配置

1.  克隆本仓库。
2.  建议使用虚拟环境：
    ```bash
    python -m venv venv
    source venv/bin/activate  # 在 Windows 上: venv\Scripts\activate
    ```
3.  从 `requirements.txt` 文件安装所需依赖。
    ```bash
    pip install -r requirements.txt
    ```

### 从源代码运行

-   **运行 GUI 应用程序**:
    ```bash
    python gui.py
    ```
-   **使用命令行界面 (CLI)**:
    ```bash
    python main.py --json <json文件路径> --pdf <pdf文件路径> --output <pptx输出路径> [OPTIONS]
    ```

### 打包为独立可执行文件 (.exe)

您可以将此 GUI 应用打包成单个 `.exe` 文件，方便分发给没有安装 Python 环境的 Windows 用户。

1.  **安装 PyInstaller**:
    ```bash
    pip install pyinstaller
    ```

2.  **构建可执行文件**:
    在项目根目录运行 `pyinstaller` 命令。使用 `--name` 参数为您的应用指定一个专业的名称。
    -   `--windowed`: 防止在运行时出现后台控制台窗口。
    -   `--onefile`: 将所有内容打包到单个可执行文件中。
    -   `--name`: 设置最终生成的可执行文件的名称。

    ```bash
    pyinstaller --windowed --onefile --name MinerU2PPT gui.py
    ```

3.  **找到可执行文件**:
    命令执行完毕后，您将在 `dist` 文件夹中找到独立的应用文件：`MinerU2PPT.exe`。

## 📚 文档

如需完整文档，请访问 **[docs/](docs/)** 文件夹：

- **[📖 完整用户指南](docs/USER_GUIDE.html)** - 支持3种语言的交互式手册
- **[🏗️ 构建说明](docs/BUILD_INSTRUCTIONS.md)** - 开发环境设置和部署
- **[🤖 AI集成指南](docs/CLAUDE.md)** - AI服务配置和使用
- **[🌍 翻译系统](docs/TRANSLATION_SYSTEM.md)** - 多语言支持详情
- **[📋 所有文档](docs/README.md)** - 完整文档索引

## 📝 更新日志

本项目的所有重要更改都记录在 **[CHANGELOG.md](CHANGELOG.md)** 中：

- **[v2.0.0 (最新版)](CHANGELOG.md#200---2026-02-10)** - AI集成、多语言支持和专业GUI的主要版本
- **[所有版本](CHANGELOG.md)** - 完整版本历史和功能添加
- **[计划功能](CHANGELOG.md#unreleased)** - 即将推出的增强功能和路线图

## 📄 许可证

本项目采用 **MIT 许可证** - 详情请见 **[LICENSE](LICENSE)** 文件。

### 致谢
这个增强版本受到 [JuniverseCoder 的原始 MinerU2PPT](https://github.com/JuniverseCoder/MinerU2PPT) 启发并在此基础上扩展。重要的增强功能（AI集成、多语言支持、专业GUI、全面文档）是在MIT许可证下的原创贡献。

### 第三方
本软件使用了各种开源库和AI服务。完整的归属和第三方致谢请参见 [LICENSE](LICENSE) 文件。

## 💖 支持

如果这个工具对您有帮助，请考虑：
- ⭐ **[为仓库加星](https://github.com/arlinamid/MinerU2PPT)**
- ☕ **[请我喝咖啡](https://buymeacoffee.com/arlinamid)**
- 🐛 **[报告问题](https://github.com/arlinamid/MinerU2PPT/issues)** 以获得改进
