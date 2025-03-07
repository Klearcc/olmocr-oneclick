
import datetime
import os
import torch
import base64
import urllib.request
import tempfile
import gradio as gr
import shutil
from io import BytesIO
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from PyPDF2 import PdfReader


from olmocr.data.renderpdf import render_pdf_to_base64png
from olmocr.prompts import build_finetuning_prompt
from olmocr.prompts.anchor import get_anchor_text




# 保存或追加内容到txt文件
def save_results_to_file(original_filename_with_timestamp, page_result):
    result_path = f"{original_filename_with_timestamp}.txt"
    with open(result_path, "a", encoding='utf-8') as f:
        f.write(f"{page_result}\n\n")
    return result_path


import datetime
import os
from PyPDF2 import PdfReader

def save_results_to_file(original_filename_with_timestamp, page_result):
    result_path = f"{original_filename_with_timestamp}.txt"
    with open(result_path, "a", encoding='utf-8') as f:
        f.write(f"{page_result}\n\n")
    return result_path

def process_single_pdf_realtime(pdf_file, temperature, max_new_tokens, num_return_sequences, do_sample):
    original_file_name = os.path.splitext(os.path.basename(pdf_file.name))[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    original_file_name_with_timestamp = f"{original_file_name}_{timestamp}"
    result_txt_path = f"{original_file_name_with_timestamp}.txt"

    if not pdf_file.name.lower().endswith(".pdf"):
        yield "", f"❌ 文件{original_file_name}非PDF格式！"
        return

    try:
        reader = PdfReader(pdf_file.name)
        num_pages = len(reader.pages)
    except Exception as e:
        yield "", f"❌ 无法读取PDF文件{original_file_name}页数: {str(e)}"
        return

    with open(result_txt_path, "w", encoding="utf-8") as f_res:
        f_res.write(f"PDF文件名称：{original_file_name}.pdf，共 {num_pages} 页\n\n")

    cumulative_errors = []

    for page_number in range(1, num_pages + 1):
        try:
            page_text, _img = process_pdf_file(
                pdf_file.name,
                page_number,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                num_return_sequences=num_return_sequences,
                do_sample=do_sample
            )

            page_output = f"==== 📄 文件【{original_file_name}.pdf】第 {page_number}/{num_pages} 页分析结果 ====\n{page_text}\n"

            save_results_to_file(original_file_name_with_timestamp, page_output)

            yield page_output, ("✅ 无错误" if not cumulative_errors else "\n\n".join(cumulative_errors))

        except Exception as page_error:
            import traceback
            error_details = traceback.format_exc()
            error_msg = f"❌ 文件【{original_file_name}.pdf】第 {page_number}/{num_pages} 页分析异常：{str(page_error)}\n详情：{error_details}\n\n"
            cumulative_errors.append(error_msg)

            save_results_to_file(original_file_name_with_timestamp, error_msg)

            yield "", error_msg

    yield f"\n📗🎉 文件【{original_file_name}.pdf】分析完成！结果保存在{result_txt_path}\n", ("✅ 无错误" if not cumulative_errors else "\n\n".join(cumulative_errors))


                     
def process_entire_file_upload_with_progress(file,
                               temperature=0.8,
                               max_new_tokens=50,
                               num_return_sequences=1,
                               do_sample=True):
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1])
        temp_file.close()
        shutil.copy(file.name, temp_file.name)

        file_extension = os.path.splitext(temp_file.name)[1].lower()

        if file_extension == ".pdf":
            # 使用上述新yield函数（带进度）
            for result, errors in process_entire_pdf_with_progress(
                            temp_file.name,
                            temperature,
                            max_new_tokens,
                            num_return_sequences,
                            do_sample,
                        ):
                yield result, errors
        else:
            yield "", "❌ 仅支持处理PDF文件，请重新上传PDF格式的文件！"

        os.unlink(temp_file.name)

    except Exception as e:
        import traceback
        yield "", f"❌ 系统错误: {str(e)}\n{traceback.format_exc()}"



def process_entire_pdf_with_progress(pdf_path, temperature=0.8, max_new_tokens=50, num_return_sequences=1, do_sample=True):
    outputs = []
    errors = []

    try:
        reader = PdfReader(pdf_path)
        num_pages = len(reader.pages)
    except Exception as e:
        yield "", f"❌获取页数时出错：{str(e)}"
        return

    for page_number in range(1, num_pages + 1):
        try:
            text_output, _ = process_pdf_file(
                pdf_path,
                page_number,
                temperature,
                max_new_tokens,
                num_return_sequences,
                do_sample,
            )
            outputs.append(f"==== 第 {page_number}/{num_pages} 页 ====\n{text_output}\n")
        except Exception as page_error:
            import traceback
            error_msg = f"第 {page_number} 页 ❌ 错误: {str(page_error)}\n详细信息:\n{traceback.format_exc()}"
            errors.append(error_msg)

        # 每处理完一页，yield输出并实时展示页面进度 
        current_progress = "\n".join(outputs)
        current_error_log = "\n\n".join(errors) if errors else "✅ 无错误"
        yield current_progress, current_error_log

    # 最后再yield一次确保完整性（处理完毕）
    final_output = "\n".join(outputs)
    final_errors = "\n\n".join(errors) if errors else "✅ 无错误"
    yield final_output, final_errors



def process_entire_file_upload(file,
                               temperature=0.8,
                               max_new_tokens=50,
                               num_return_sequences=1,
                               do_sample=True):
    try:
        # 临时保存PDF文件
        temp_file = tempfile.NamedTemporaryFile(
            delete=False, suffix=os.path.splitext(file.name)[1]
        )
        temp_file.close()
        shutil.copy(file.name, temp_file.name)

        # 确认文件扩展名
        file_extension = os.path.splitext(temp_file.name)[1].lower()

        if file_extension == ".pdf":
            # 处理整个PDF，并捕获错误
            result, errors = process_entire_pdf(
                temp_file.name,
                temperature,
                max_new_tokens,
                num_return_sequences,
                do_sample,
            )
        else:
            result = ""
            errors = "❌ 仅支持PDF文件，请重新上传正确格式的文件！"

        os.unlink(temp_file.name)

        # 如果无错误信息，返回"✅ 无错误"
        error_output = errors if errors else "✅ 无错误"

        return result, error_output

    except Exception as e:
        import traceback
        return "", f"❌系统错误: {str(e)}\n{traceback.format_exc()}"
    

def process_entire_pdf(pdf_path, temperature=0.8, max_new_tokens=50, num_return_sequences=1, do_sample=True):
    outputs = []  # 用来累积每页输出
    errors = []   # 用来存储每页可能发生的错误信息

    # 首先获取PDF文件页数
    try:
        reader = PdfReader(pdf_path)
        num_pages = len(reader.pages)
    except Exception as e:
        return "", f"获取页数时出错：{str(e)}"

    for page_number in range(1, num_pages + 1):
        try:
            # 调用你原来已有的单页处理函数
            text_output, _ = process_pdf_file(
                pdf_path,
                page_number,
                temperature,
                max_new_tokens,
                num_return_sequences,
                do_sample,
            )
            outputs.append(f"===== 第 {page_number} 页 =====\n{text_output}\n")
        except Exception as page_error:
            import traceback
            error_msg = f"第 {page_number} 页处理出错: {str(page_error)}\n详情:\n{traceback.format_exc()}"
            errors.append(error_msg)

    # 拼接所有输出
    final_output = "\n\n".join(outputs)

    # 拼接所有的错误信息
    final_errors = "\n\n".join(errors)

    return final_output, final_errors


# Initialize the model (globally to avoid reloading it on each request)
print("Initializing the model...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "allenai/olmOCR-7B-0225-preview", torch_dtype=torch.bfloat16
).eval()
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct",use_fast=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Model loaded on {device}")


def download_pdf(url):
    """Download a PDF from the specified URL"""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    try:
        urllib.request.urlretrieve(url, temp_file.name)
        return temp_file.name
    except Exception as e:
        return None, f"Error during PDF download: {str(e)}"


def process_pdf_url(
    url,
    page_number=1,
    temperature=0.8,
    max_new_tokens=4096,
    num_return_sequences=1,
    do_sample=True,
):
    """Process a PDF from URL and generate output using olmOCR"""
    try:
        # Download the PDF
        pdf_path = download_pdf(url)
        if pdf_path is None:
            return "Error downloading the PDF", None

        # Process the PDF
        result, image = process_pdf_file(
            pdf_path,
            page_number,
            temperature,
            max_new_tokens,
            num_return_sequences,
            do_sample,
        )

        # Clean up temporary files
        os.unlink(pdf_path)

        return result, image

    except Exception as e:
        import traceback

        return f"Error: {str(e)}\n{traceback.format_exc()}", None


def process_pdf_file(
    pdf_path,
    page_number=1,
    temperature=0.8,
    max_new_tokens=4096,
    num_return_sequences=1,
    do_sample=True,
):
    """Process a local PDF and generate output using olmOCR"""
    try:
        # Render the PDF page as an image
        image_base64 = render_pdf_to_base64png(
            pdf_path, page_number, target_longest_image_dim=1024
        )

        # Process the PDF with the generated base64
        return process_pdf_base64(
            image_base64,
            pdf_path,
            page_number,
            temperature,
            max_new_tokens,
            num_return_sequences,
            do_sample,
        )

    except Exception as e:
        import traceback

        return f"Error: {str(e)}\n{traceback.format_exc()}", None


def process_file_upload(
    file,
    page_number=1,
    temperature=0.8,
    max_new_tokens=50,
    num_return_sequences=1,
    do_sample=True,
):
    """Process a file (PDF or image) uploaded by the user"""
    try:
        # Save the uploaded file temporarily
        temp_file = tempfile.NamedTemporaryFile(
            delete=False, suffix=os.path.splitext(file.name)[1]
        )
        temp_file.close()
        shutil.copy(file.name, temp_file.name)

        # Determine if it's a PDF or an image
        file_extension = os.path.splitext(temp_file.name)[1].lower()

        if file_extension == ".pdf":
            # Process as PDF
            result, image = process_pdf_file(
                temp_file.name,
                page_number,
                temperature,
                max_new_tokens,
                num_return_sequences,
                do_sample,
            )
        elif file_extension in [
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".tiff",
            ".tif",
            ".webp",
        ]:
            # Process as image
            result, image = process_image_file(
                temp_file.name,
                temperature,
                max_new_tokens,
                num_return_sequences,
                do_sample,
            )
        else:
            result = f"Unsupported file format: {file_extension}. Please use PDF or images (JPG, PNG, etc.)"
            image = None

        # Clean up temporary files
        os.unlink(temp_file.name)

        return result, image

    except Exception as e:
        import traceback

        return f"Error: {str(e)}\n{traceback.format_exc()}", None


def process_image_file(
    image_path,
    temperature=0.8,
    max_new_tokens=50,
    num_return_sequences=1,
    do_sample=True,
):
    """Process a local image and generate output using olmOCR"""
    try:
        # Open the image
        with open(image_path, "rb") as img_file:
            img_data = img_file.read()
            image_base64 = base64.b64encode(img_data).decode("utf-8")

        # Use a generic anchor text for images
        anchor_text = "Image analysis."

        # Process the image with the generated base64
        return process_pdf_base64(
            image_base64,
            None,  # No PDF path
            1,  # Not applicable for images
            temperature,
            max_new_tokens,
            num_return_sequences,
            do_sample,
            anchor_text=anchor_text,
        )

    except Exception as e:
        import traceback

        return f"Error: {str(e)}\n{traceback.format_exc()}", None


def process_pdf_base64(
    image_base64,
    pdf_path=None,
    page_number=1,
    temperature=0.8,
    max_new_tokens=50,
    num_return_sequences=1,
    do_sample=True,
    anchor_text=None,
):
    """Process an image in base64 format and generate output using olmOCR"""
    try:
        # If a PDF path was provided, get the anchor text
        if pdf_path and not anchor_text:
            anchor_text = get_anchor_text(
                pdf_path, page_number, pdf_engine="pdfreport", target_length=4000
            )
        elif not anchor_text:
            # If we don't have a PDF or a specified anchor text, use a generic anchor text
            anchor_text = "Document analysis."

        prompt = build_finetuning_prompt(anchor_text)

        # Build the complete prompt
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                    },
                ],
            }
        ]

        # Apply the chat template and processor
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        main_image = Image.open(BytesIO(base64.b64decode(image_base64)))

        # Display the image
        rendered_image = main_image.copy()

        inputs = processor(
            text=[text],
            images=[main_image],
            padding=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(device) for (key, value) in inputs.items()}

        # Generate the output
        output = model.generate(
            **inputs,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            num_return_sequences=num_return_sequences,
            do_sample=do_sample,
        )

        # Decode the output
        prompt_length = inputs["input_ids"].shape[1]
        new_tokens = output[:, prompt_length:]
        text_output = processor.tokenizer.batch_decode(
            new_tokens, skip_special_tokens=True
        )

        return text_output[0], rendered_image

    except Exception as e:
        import traceback

        return f"Error: {str(e)}\n{traceback.format_exc()}", None


# Create the Gradio interface
with gr.Blocks(title="olmOCR Document Analyzer") as demo:
    gr.Markdown("# olmOCR Document Analyzer")
    gr.Markdown("Analyze PDF documents and images using the olmOCR-7B model")

    with gr.Tabs() as tabs:
        with gr.TabItem("PDF URL"):
            with gr.Row():
                with gr.Column(scale=2):
                    url_input = gr.Textbox(
                        label="PDF URL",
                        placeholder="https://example.com/document.pdf",
                    )
                    page_number_url = gr.Number(
                        label="Page Number", value=1, minimum=1, step=1
                    )

                    with gr.Row():
                        with gr.Column():
                            temperature_url = gr.Slider(
                                label="Temperature",
                                minimum=0.0,
                                maximum=1.0,
                                value=0.8,
                                step=0.1,
                            )
                            max_new_tokens_url = gr.Slider(
                                label="Max New Tokens",
                                minimum=10,
                                maximum=4096,
                                value=1024,
                                step=10,
                            )

                        with gr.Column():
                            num_return_sequences_url = gr.Slider(
                                label="Number of Returned Sequences",
                                minimum=1,
                                maximum=5,
                                value=1,
                                step=1,
                            )
                            do_sample_url = gr.Checkbox(label="Do Sample", value=True)

                    submit_btn_url = gr.Button("Analyze PDF", variant="primary")

                with gr.Column(scale=3):
                    with gr.Row():
                        with gr.Column():
                            image_output_url = gr.Image(label="PDF Page", type="pil")

                        with gr.Column():
                            text_output_url = gr.Textbox(label="Result", lines=10)

                submit_btn_url.click(
                    fn=process_pdf_url,
                    inputs=[
                        url_input,
                        page_number_url,
                        temperature_url,
                        max_new_tokens_url,
                        num_return_sequences_url,
                        do_sample_url,
                    ],
                    outputs=[text_output_url, image_output_url],
                )

                gr.Markdown("### Example URL")
                gr.Examples(
                    examples=[
                        ["https://molmo.allenai.org/paper.pdf", 1, 0.8, 50, 1, True],
                    ],
                    inputs=[
                        url_input,
                        page_number_url,
                        temperature_url,
                        max_new_tokens_url,
                        num_return_sequences_url,
                        do_sample_url,
                    ],
                )

        with gr.TabItem("处理多个PDF文件（逐页实时分析+保存结果）"):
            with gr.Row():
                with gr.Column(scale=2):
                    files_input = gr.Files(
                        label="上传多个PDF文件",
                        file_types=[".pdf"]
                    )

                    with gr.Row():
                        with gr.Column():
                            temperature_file = gr.Slider(0, 1, value=0.8, step=0.1, label="Temperature")
                            max_new_tokens_file = gr.Slider(10, 4096, value=1024, step=10, label="Max New Tokens")

                        with gr.Column():
                            num_return_sequences_file = gr.Slider(1, 5, value=1, step=1, label="Num Returned Sequences")
                            do_sample_file = gr.Checkbox(value=True, label="Do Sample")

                    submit_btn_files = gr.Button("开始分析多个PDF文件", variant="primary")

                with gr.Column(scale=3):
                    realtime_result = gr.Textbox(label="📄 每个文件实时逐页分析结果", lines=25, max_lines=100, interactive=False)
                    realtime_errors = gr.Textbox(label="⚠️ 错误日志实时提示", lines=10, interactive=False, value="✅ 无错误")

                    def analyze_multiple_pdfs(files, temperature, max_new_tokens, num_return_sequences, do_sample):
                        for pdf_file in files:
                            # 开始分析提示信息
                            yield f"\n\n📚✨ 开始分析文件【{pdf_file.name}】\n", "✅ 无错误"
                            
                            # 对单个pdf文件逐页处理并实时输出单页最新结果
                            for page_output, error_info in process_single_pdf_realtime(
                                pdf_file, temperature, max_new_tokens, num_return_sequences, do_sample
                            ):
                                # 每处理完一页立即yield单页分析结果，不要累积全部历史页内容，避免前端卡顿
                                yield page_output, error_info
                            
                            # 文件处理完成后的提示信息
                            yield f"\n📗🎉 文件【{pdf_file.name}】分析完成！\n", error_info

                    # Gradio的click函数替换：
                    submit_btn_files.click(
                        fn=analyze_multiple_pdfs,
                        inputs=[files_input, temperature_file, max_new_tokens_file, num_return_sequences_file, do_sample_file],
                        outputs=[realtime_result, realtime_errors]
                    )

        with gr.TabItem("Direct Base64"):
            with gr.Row():
                with gr.Column(scale=2):
                    base64_input = gr.Textbox(
                        label="Enter the base64 string of the image", lines=5
                    )

                    with gr.Row():
                        with gr.Column():
                            temperature_base64 = gr.Slider(
                                label="Temperature",
                                minimum=0.0,
                                maximum=1.0,
                                value=0.8,
                                step=0.1,
                            )
                            max_new_tokens_base64 = gr.Slider(
                                label="Max New Tokens",
                                minimum=10,
                                maximum=4096,
                                value=1024,
                                step=10,
                            )

                        with gr.Column():
                            num_return_sequences_base64 = gr.Slider(
                                label="Number of Returned Sequences",
                                minimum=1,
                                maximum=5,
                                value=1,
                                step=1,
                            )
                            do_sample_base64 = gr.Checkbox(
                                label="Do Sample", value=True
                            )

                    submit_btn_base64 = gr.Button("Analyze Image", variant="primary")

                with gr.Column(scale=3):
                    with gr.Row():
                        with gr.Column():
                            image_output_base64 = gr.Image(
                                label="Decoded Image", type="pil"
                            )

                        with gr.Column():
                            text_output_base64 = gr.Textbox(label="Result", lines=10)

                submit_btn_base64.click(
                    fn=lambda b64, t, m, n, d: process_pdf_base64(
                        b64, None, 1, t, m, n, d
                    ),
                    inputs=[
                        base64_input,
                        temperature_base64,
                        max_new_tokens_base64,
                        num_return_sequences_base64,
                        do_sample_base64,
                    ],
                    outputs=[text_output_base64, image_output_base64],
                )

# Launch the app
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=True)
