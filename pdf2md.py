import logging
import time
import base64
import json
import yaml
import hashlib
from pathlib import Path
from io import BytesIO
import hashlib
from pdf2image import convert_from_path

import pandas as pd
from PIL import Image
from openai import OpenAI

from docling_core.types.doc import PictureItem, TableItem
from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from prompt.VLM_prompt import VLM_PROMPT
from prompt.text_type_prompt import TEXT_TYPE_PROMPT
from prompt.table_repair_prompt import TABLE_REPAIR_PROMPT


# 加载配置文件
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# === 配置 === (从配置文件读取)
ENABLE_OCR = config['OCR']['enabled']
DASHSCOPE_API_KEY = config['VLM']['api_key']
VLM_API_URL = config['VLM']['base_url']
VLM_MODEL = config['VLM']['model']

TEXT_API_KEY = config['OPENAI']['api_key']
TEXT_API_URL = config['OPENAI']['base_url']
TEXT_MODEL = config['OPENAI']['model']

input_pdf_path = Path("D:/Personal_Project/SmolDocling/pdfs/APD_Series_203250D.pdf")

# 根据 PDF 文件路径生成哈希值
def generate_hash_from_file(file_path: Path) -> str:
    md5_hash = hashlib.md5()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()

# 获取哈希值作为子目录名
pdf_hash = generate_hash_from_file(input_pdf_path)
output_dir = Path.cwd() / "output" / pdf_hash
output_dir.mkdir(parents=True, exist_ok=True)
doc_filename = input_pdf_path.stem

# === 获取PDF每页并保存为图片 ===
def convert_pdf_to_images(pdf_path: Path, output_dir: Path):
    # 从配置中获取 Poppler 路径
    poppler_path = Path(config['POPPLER']['path'])

    # 创建 page 子目录
    page_dir = output_dir / "page"
    page_dir.mkdir(parents=True, exist_ok=True)

    # 使用 pdf2image 将每一页转换为图片
    pages = convert_from_path(
        pdf_path,
        dpi=300,  # 300 DPI
        poppler_path=str(poppler_path)
    )

    for page_num, page in enumerate(pages, start=1):
        page_image_filename = page_dir / f"page-{page_num}.png"
        page.save(page_image_filename, 'PNG')
        log.info(f"保存 PDF 第 {page_num} 页：{page_image_filename.resolve()}")

# === 图像 + Prompt → Markdown 表格（Qwen）===
def ask_table_from_image(pil_image: Image.Image, prompt: str = TABLE_REPAIR_PROMPT) -> str:
    try:
        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        client = OpenAI(api_key=DASHSCOPE_API_KEY, base_url=VLM_API_URL)
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
        ]
        completion = client.chat.completions.create(
            model=VLM_MODEL,
            messages=[{"role": "user", "content": content}]
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        log.warning(f"❌ 表格图像修复失败: {e}")
        return "[表格修复失败]"


# === 图片描述 ===
def ask_image_vlm_base64(pil_image: Image.Image, prompt: str = VLM_PROMPT) -> str:
    try:
        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        client = OpenAI(api_key=DASHSCOPE_API_KEY, base_url=VLM_API_URL)
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
        ]
        completion = client.chat.completions.create(
            model="qwen-vl-plus",
            messages=[{"role": "user", "content": content}]
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        log.warning(f"图像API失败: {e}")
        return "[图像描述失败]"


# === 判断文本类型（标题 or 正文）===
def ask_if_heading(text: str) -> str:
    try:
        client = OpenAI(api_key=TEXT_API_KEY, base_url=TEXT_API_URL)
        prompt = f"{TEXT_TYPE_PROMPT}\n{text}"
        response = client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response.choices[0].message.content.strip().lower()
        return "heading" if "heading" in answer else "paragraph"
    except Exception as e:
        log.warning(f"判断标题/正文失败: {e}")
        return "paragraph"


# === 表格图像切块工具（按固定行高裁切） ===
def split_table_image_rows(pil_img: Image.Image, row_height: int = 400) -> list:
    width, height = pil_img.size
    slices = []
    for top in range(0, height, row_height):
        bottom = min(top + row_height, height)
        crop = pil_img.crop((0, top, width, bottom))
        slices.append(crop)
    return slices

# === 获取元素的边界框 ===
def get_bbox(element):
    if hasattr(element, 'prov') and element.prov:
        bbox = element.prov[0].bbox
        return {
            "left": bbox.l,
            "top": bbox.t,
            "right": bbox.r,
            "bottom": bbox.b,
            "coord_origin": bbox.coord_origin
        }
    return None

# === 主流程 ===
def convert_pdf_to_markdown_with_images():
    start_time = time.time()
    pipeline_options = PdfPipelineOptions()
    pipeline_options.images_scale = 2.0
    pipeline_options.generate_picture_images = True
    pipeline_options.generate_table_images = True

    if ENABLE_OCR:
        pipeline_options.do_ocr = True
        pipeline_options.ocr_options = RapidOcrOptions(force_full_page_ocr=True)

    doc_converter = DocumentConverter(
        format_options={"pdf": PdfFormatOption(pipeline_options=pipeline_options)}
    )
    conv_res = doc_converter.convert(input_pdf_path)
    document = conv_res.document

    markdown_lines = []
    json_data = []
    table_counter = 0
    picture_counter = 0

    # 获取并保存 PDF 每一页为图片
    convert_pdf_to_images(input_pdf_path, output_dir)

    for element, level in document.iterate_items():
        # 获取元素的边界框
        bbox = get_bbox(element)

        if isinstance(element, TableItem):
            table_counter += 1
            # 使用哈希值生成图片/表格文件名
            table_image_filename = output_dir / f"{pdf_hash}-table-{table_counter}.png"
            pil_img = element.get_image(document)
            pil_img.save(table_image_filename, "PNG")
            table_df: pd.DataFrame = element.export_to_dataframe()

            if not table_df.columns.is_unique or table_df.shape[1] < 2:
                log.warning(f"⚠️ 表格 {table_counter} 结构异常，使用 Qwen 多轮图像推理修复")

                # 自动图像切块
                sub_images = split_table_image_rows(pil_img)
                all_chunks = []

                for idx, chunk_img in enumerate(sub_images):
                    chunk_md = ask_table_from_image(chunk_img)
                    all_chunks.append(chunk_md.strip())

                # 拼接多段 Markdown 表格（保留首段表头）
                full_md_lines = []
                for i, chunk in enumerate(all_chunks):
                    lines = chunk.splitlines()
                    if i == 0:
                        full_md_lines.extend(lines)  # 保留表头 + 分割线
                    else:
                        full_md_lines.extend(lines[2:])  # 仅添加数据行

                markdown_lines.append(f"\n<!-- 表格 {table_counter} 使用 Qwen 修复，已分块拼接 -->\n")
                markdown_lines.append("\n".join(full_md_lines))
                markdown_lines.append("")
                json_data.append({
                    "type": "table",
                    "level": level,
                    "image": table_image_filename.name,
                    "source": "reconstructed_by_qwen_chunked",
                    "markdown": "\n".join(full_md_lines),
                    "page_number": element.prov[0].page_no,  # Add page number to JSON
                    "bbox": bbox  # 添加边界框到 JSON
                })
                continue  # 跳过原始处理

            # ✅ 表格结构正常
            markdown_lines.append(table_df.to_markdown(index=False))
            markdown_lines.append("")
            json_data.append({
                "type": "table",
                "level": level,
                "image": table_image_filename.name,
                "data": table_df.to_dict(orient="records"),
                "page_number": element.prov[0].page_no,  # Add page number to JSON
                "bbox": bbox  # 添加边界框到 JSON
            })

        elif isinstance(element, PictureItem):
            picture_counter += 1
            # 使用哈希值生成图片文件名
            picture_image_filename = output_dir / f"{pdf_hash}-picture-{picture_counter}.png"
            pil_img = element.get_image(document)
            pil_img.save(picture_image_filename, "PNG")
            caption = ask_image_vlm_base64(pil_img)

            # 生成图片链接 + 描述为图片标题
            markdown_lines.append(f"\n![{caption}](./{picture_image_filename.name})\n")
            json_data.append({
                "type": "picture",
                "level": level,
                "image": picture_image_filename.name,
                "caption": caption,
                "page_number": element.prov[0].page_no,  # Add page number to JSON
                "bbox": bbox  # 添加边界框到 JSON
            })

        else:
            if hasattr(element, "text") and element.text:
                text = element.text.strip()
                if text:
                    label = ask_if_heading(text)
                    markdown_lines.append(f"# {text}" if label == "heading" else text)
                    markdown_lines.append("")
                    json_data.append({
                        "type": "text",
                        "level": level,
                        "text": text,
                        "label": label,
                        "page_number": element.prov[0].page_no,  # Add page number to JSON
                        "bbox": bbox  # 添加边界框到 JSON
                    })

    # 保存结果
    markdown_file = output_dir / f"{pdf_hash}.md"
    with markdown_file.open("w", encoding="utf-8") as f:
        f.write("\n".join(markdown_lines))

    json_file = output_dir / f"{pdf_hash}.json"
    with json_file.open("w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    log.info(f"✅ 完成 PDF 解析，耗时 {time.time() - start_time:.2f} 秒")
    log.info(f"📄 Markdown 文件：{markdown_file.resolve()}")
    log.info(f"📦 JSON 文件：{json_file.resolve()}")

if __name__ == "__main__":
    convert_pdf_to_markdown_with_images()
