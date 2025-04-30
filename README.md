# Docling2md

将 PDF 文档结构化为 Markdown 和 JSON 格式的轻量级工具，支持表格修复、图像描述、标题识别和结构化抽取。

---

## 项目简介
**Docling2md** 是一个文档理解和转换工具，基于 [Docling](https://github.com/Docling/Docling) 框架，并集成了视觉语言模型（VLM）和大语言模型（LLM），可实现：

- 将 PDF 转换为结构化 Markdown 文本
- 自动提取标题 / 正文结构
- 使用 VLM 自动描述图片
- 修复表格图像识别失败问题（表格切片 + Qwen-VL 修复）
- 输出结构化 JSON，便于下游知识图谱、搜索索引等任务

---

## 安装与环境配置

### 1. 克隆仓库
```bash
git clone https://github.com/your-username/Docling2md.git
cd Docling2md
```

### 2. 安装依赖
推荐使用 `conda` 创建虚拟环境：
```bash
conda create -n docling2md python=3.10 -y
conda activate docling2md
pip install -r requirements.txt
```

解压 `poppler` 压缩包至项目根目录。

### 3. 配置文件（`config.yaml`）
```yaml
OPENAI:
  api_key: "<your-deepseek-api-key>"
  base_url: "https://api.deepseek.com"
  model: "deepseek-chat"  #默认使用deepseek的chat模型
  max_concurrency: 10  #chat大模型最大并发量

VLM:
  api_key: "<your-dashscope-api-key>"
  base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
  model: "qwen2.5-vl-72b-instruct"  #默认使用qwen视觉模型
  max_concurrency: 3  #视觉大模型最大并发量

OCR:
  enabled: true

POPPLER:
  path: "D:/your_path_to/poppler/Library/bin"
```

---

## 使用方法

### 1. 修改读取的PDF文件路径
```bash
input_pdf_path = Path("D:/Docling2md/your_path_to/.pdf")
```

### 2. 运行主程序
```bash
python pdf2md.py
```
或将 `convert_pdf_to_markdown_with_images()` 嵌入你的主流程中。

### 3. 结果输出
- Markdown 文件：`output/<pdf_hash>/<hash>.md`
- JSON 文件：`output/<pdf_hash>/<hash>.json`
- 页面图像：`output/<pdf_hash>/page/page-*.png`
- 表格与图片图像：`output/<pdf_hash>/*.png`

---

## 功能亮点

- **PDF 布局解析**：使用 Docling 提取表格、图片、文本等元素
- **表格图像修复**：支持表格列名冲突自动切片 + VLM 重建
- **图像描述理解**：VLM 自动输出图片标题并写入 Markdown
- **文本分类**：调用 DeepSeek 判断是否为标题，决定 Markdown 格式
- **结构化输出**：统一 JSON 格式，包含 `type`, `level`, `page_number`, `bbox` 等字段

---

## 项目结构
```
.
├── pdf2md.py
├── config.yaml
├── prompt/
│   ├── VLM_prompt.py
│   ├── text_type_prompt.py
│   └── table_repair_prompt.py
├── output/
│   └── <pdf_hash>/
│       ├── page/
│       ├── *.png
│       ├── *.md
│       └── *.json
└── ...
```

---

## 使用视觉模型修复表格示例效果展示

![image](https://github.com/user-attachments/assets/fe9488f9-6a5f-46a0-99b1-a170b1819d19)

```
| Part Number.Part Number   | Function.Function                       |   Output.Channel | Characteristics.V DD(opr) (V)   | Characteristics.Output Current   | Characteristics.T opr (°C)   | Package.Package   | Protection.Over Current   | Protection.Over Voltage   | Protection.Under Voltage   | Protection.Reverse Battery   | Diagnosis.Over Current   | Diagnosis.Over Voltage   | Diagnosis.Under Voltage   | AEC-Qxxx qualified.AEC-Qxxx qualified   |
|:--------------------------|:----------------------------------------|-----------------:|:--------------------------------|:---------------------------------|:-----------------------------|:------------------|:--------------------------|:--------------------------|:---------------------------|:-----------------------------|:-------------------------|:-------------------------|:--------------------------|:----------------------------------------|
| TPD7104AF                 | High-side N-ch MOSFET Gate driver       |                1 | 5 to 18                         | Depends on internal drive        | -40 to 125                   | PS-8              | Adjust- able              | -                         | -                          | ✓                            | ✓                        | - -                      |                           | AEC-Q100                                |
| TPD7106F                  | High-side N-ch MOSFET Gate driver       |                1 | 4.5 to 27                       | OUT1:+/-10 mA OUT2:0. 4 A (Sink) | -40 to 150                   | SSOP-16           | -                         | -                         | -                          | ✓                            | -                        | -                        | -                         | AEC-Q100                                |
| TPD7107F                  | High-side N-ch MOSFET Gate driver       |                1 | 5.75 to 26                      | Source:100 μA Sink:5 mAor 270 mA | -40 to 125                   | WSON10A           | ✓                         | ✓                         | ✓                          | ✓                            | ✓                        | ✓                        | ✓                         | AEC-Q100                                |
| TPD7211F                  | Half-bridge MOSFET Gate driver          |                2 | 5 to 18                         | ±0.5 A max                       | -40 to 125                   | PS-8              | -                         | -                         | -                          | -                            | -                        | -                        | -                         | -                                       |
| TPD7213FN*                | Half-bridge MOSFET Gate driver          |                2 | V DD 9 to 18 V DD48 12 to 54    | Source:2 A Sink:2 A              | -40 to 150                   | SSOP-16           | -                         | -                         | ✓                          | -                            | -                        | -                        | ✓                         | **                                      |
| TPD7212F                  | 3 Phase Full- bridge MOSFET Gate driver |                6 | 4.5 to 18                       | Source:1 A Sink:1.5 A            | -40 to 150                   | WQFN32            | -                         | ✓                         | ✓                          | -                            | -                        | ✓                        | ✓                         | AEC-Q100                                |
| TPD7212FN                 | 3 Phase Full- bridge MOSFET Gate driver |                6 | 4.5 to 18                       | Source:1 A Sink:1.5 A            | -40 to 150                   | SSOP30            | -                         | ✓                         | ✓                          | -                            | -                        | ✓                        | ✓                         | AEC-Q100                                |

```

---

## 联系方式
如有建议或问题，欢迎通过 issue 或 PR 提交反馈。

---

## License
本项目遵循 MIT 协议。

---

> 本项目集成 Qwen-VL 与 DeepSeek 模型，仅供学术研究与技术验证用途。

