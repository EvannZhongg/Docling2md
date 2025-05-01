import logging
import time
import base64
import json
import yaml
import hashlib
import re
from pathlib import Path
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from prompt.text_repair_prompt import TEXT_REPAIR_PROMPT

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
MAX_CONCURRENCY_VLM = config['VLM']['max_concurrency']
MAX_CONCURRENCY_TEXT = config['OPENAI']['max_concurrency']

input_pdf_path = Path("D:/Personal_Project/SmolDocling/pdfs/Gan.pdf")


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
            model=VLM_MODEL,
            messages=[{"role": "user", "content": content}]
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        log.warning(f"图像API失败: {e}")
        return "[图像描述失败]"

def needs_repair(text: str, threshold: int = 30) -> bool:
    # 匹配连续的纯英文字符串（不包含中文或空格）
    matches = re.findall(r'[A-Za-z0-9,.\-()]{%d,}' % threshold, text)
    return len(matches) > 0


# 大模型进行英文分词修复
def ask_repair_text(text: str) -> str:
    try:
        client = OpenAI(api_key=TEXT_API_KEY, base_url=TEXT_API_URL)
        prompt = f"{TEXT_REPAIR_PROMPT}\n{text}"
        response = client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        repaired = response.choices[0].message.content.strip()
        return repaired
    except Exception as e:
        log.warning(f"❌ 英文分词失败: {e}")
        return text  # 失败时返回原文

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


# === 拼接不符合尺寸限制的切块 ===
def merge_small_chunks(chunks: list, min_height: int = 300, min_width: int = 20) -> list:
    merged_chunks = []
    temp_chunk = None

    for chunk in chunks:
        width, height = chunk.size

        # 如果当前块尺寸不足，则尝试拼接上下块
        if height < min_height or width < min_width:
            if temp_chunk is None:
                temp_chunk = chunk
            else:
                # 拼接上下块
                new_chunk = Image.new("RGB", (max(temp_chunk.width, chunk.width), temp_chunk.height + chunk.height))
                new_chunk.paste(temp_chunk, (0, 0))
                new_chunk.paste(chunk, (0, temp_chunk.height))
                temp_chunk = new_chunk
        else:
            # 如果有未处理的临时块，先保存
            if temp_chunk is not None:
                merged_chunks.append(temp_chunk)
                temp_chunk = None
            merged_chunks.append(chunk)

    # 添加最后一个临时块（如果有）
    if temp_chunk is not None:
        # 如果整个表格图片的高度低于最小高度，则按最低高度计算
        if temp_chunk.height < min_height:
            new_chunk = Image.new("RGB", (temp_chunk.width, max(temp_chunk.height, 20)))
            new_chunk.paste(temp_chunk, (0, 0))
            merged_chunks.append(new_chunk)
        else:
            merged_chunks.append(temp_chunk)

    return merged_chunks


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

    # 并发任务队列
    vlm_executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENCY_VLM)
    text_executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENCY_TEXT)

    futures = []

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
                # 拼接不符合尺寸限制的切块
                sub_images = merge_small_chunks(sub_images)

                # 🔁 使用局部变量管理该表格的修复任务
                chunk_futures = []
                for idx, chunk_img in enumerate(sub_images):
                    future = vlm_executor.submit(ask_table_from_image, chunk_img)
                    chunk_futures.append((future, idx, chunk_img))

                # 收集结果
                full_md_lines = []
                for future, idx, chunk_img in chunk_futures:
                    try:
                        chunk_md = future.result()
                        lines = chunk_md.splitlines()
                        if idx == 0:
                            full_md_lines.extend(lines)  # 表头 + 分隔线
                        else:
                            full_md_lines.extend(lines[2:])  # 仅数据行
                    except Exception as e:
                        log.warning(f"表格分块处理失败: {e}")

                markdown_lines.append(f"<!-- 表格 {table_counter} 使用 Qwen 修复，已分块拼接 -->")
                markdown_lines.append("\n".join(full_md_lines))
                markdown_lines.append("")
                json_data.append({
                    "type": "table",
                    "level": level,
                    "image": table_image_filename.name,
                    "source": "reconstructed_by_qwen_chunked",
                    "markdown": "\n".join(full_md_lines),
                    "page_number": element.prov[0].page_no,
                    "bbox": bbox
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
                "page_number": element.prov[0].page_no,
                "bbox": bbox
            })
        elif isinstance(element, PictureItem):
            picture_counter += 1
            # 使用哈希值生成图片文件名
            picture_image_filename = output_dir / f"{pdf_hash}-picture-{picture_counter}.png"
            pil_img = element.get_image(document)
            pil_img.save(picture_image_filename, "PNG")
            future = vlm_executor.submit(ask_image_vlm_base64, pil_img)
            futures.append((future, "picture", {
                "image_path": picture_image_filename,
                "level": level,
                "page": element.prov[0].page_no,
                "bbox": bbox
            }))
        else:
            if hasattr(element, "text") and element.text:
                text = element.text.strip()
                if text:
                    if needs_repair(text):
                        log.info(f"发现异常无空格段，调用分词模型修复: {text}")
                        text = ask_repair_text(text)

                    future = text_executor.submit(ask_if_heading, text)
                    futures.append((future, "text", {
                        "text": text,
                        "level": level,
                        "page": element.prov[0].page_no,
                        "bbox": bbox
                    }))

    # 等待所有并发任务完成
    for future, task_type, meta in futures:
        try:
            result = future.result()
            if task_type == "picture":
                caption = result
                markdown_lines.append(f"![{caption}](./{meta['image_path'].name})")
                json_data.append({
                    "type": "picture",
                    "level": meta["level"],
                    "image": meta["image_path"].name,
                    "caption": caption,
                    "page_number": meta["page"],
                    "bbox": meta["bbox"]
                })
            elif task_type == "text":
                label = result
                markdown_lines.append(f"# {meta['text']}" if label == "heading" else meta["text"])
                markdown_lines.append("")
                json_data.append({
                    "type": "text",
                    "level": meta["level"],
                    "text": meta["text"],
                    "label": label,
                    "page_number": meta["page"],
                    "bbox": meta["bbox"]
                })
        except Exception as e:
            log.warning(f"并发任务失败: {e}")

    # 关闭线程池
    vlm_executor.shutdown(wait=True)
    text_executor.shutdown(wait=True)

    # 保存结果
    markdown_file = output_dir / f"{pdf_hash}.md"
    with markdown_file.open("w", encoding="utf-8") as f:
        f.write("\n".join(markdown_lines))
    json_file = output_dir / f"{pdf_hash}.json"
    with json_file.open("w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    log.info(f"完成 PDF 解析，耗时 {time.time() - start_time:.2f} 秒")
    log.info(f"Markdown 文件：{markdown_file.resolve()}")
    log.info(f"JSON 文件：{json_file.resolve()}")


if __name__ == "__main__":
    convert_pdf_to_markdown_with_images()
