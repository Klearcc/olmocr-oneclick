import os
import datetime
import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from olmocr.data.renderpdf import render_pdf_to_base64png
from olmocr.prompts import build_finetuning_prompt, anchor as anchor_utils
from PyPDF2 import PdfReader
from PIL import Image
from io import BytesIO
import base64
import logging

# 配置日志系统：
logging.basicConfig(filename='processing_log.log', level=logging.INFO,format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# 单例模式管理模型生命周期：
class ModelManager:
    _instance = None

    @staticmethod
    def get_model():
        if ModelManager._instance is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                "allenai/olmOCR-7B-0225-preview",
                torch_dtype=torch.bfloat16).to(device).eval()
            
            processor = AutoProcessor.from_pretrained(
                "Qwen/Qwen2-VL-7B-Instruct", use_fast=True)

            logger.info(f"Model loaded successfully to {device}")
            
            ModelManager._instance = (model, processor, device)
        return ModelManager._instance


def process_pdf_batch(pdf_path, page_numbers, temperature=0.4,max_new_tokens=4096):
    model, processor, device = ModelManager.get_model()
    
    batch_texts,batch_images=[],[]
    
    for page_number in page_numbers:
        try:
            print(f"🔍 开始处理第 {page_number} 页")
            
            image_base64=render_pdf_to_base64png(pdf_path,page_number,target_longest_image_dim=1024)
            print(f"✅ 第 {page_number} 页图片渲染完成")
            
            anchor_text=anchor_utils.get_anchor_text(pdf_path,page_number,pdf_engine="pdfreport",target_length=4000)
            print(f"✅ 第 {page_number} 页 Anchor 文本抽取完成")

            prompt=build_finetuning_prompt(anchor_text)
            
            messages=[
                {
                    "role":"user",
                    "content":[
                        {"type":"text","text":prompt},
                        {"type":"image_url","image_url":{"url":f"data:image/png;base64,{image_base64}"}}
                    ]
                }
            ]

            text_input=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            
            image_obj_pil = Image.open(BytesIO(base64.b64decode(image_base64)))
        
        except Exception as e:
            error_detail=f"[Page {page_number}] Preparation failed:{e}"
            
            logger.warning(error_detail) 
            
            # 添加此行以便即时查看错误：
            print(error_detail)

            text_input=""
          
           # 为避免错误中断后续推理，这里创建简单空白图像。
           # 注意实际场景中可能需跳过，视需求而定！
            image_obj_pil = Image.new('RGB',(224,224),'white')

        batch_texts.append(text_input)
        batch_images.append(image_obj_pil)

    try:
        inputs_dict_tensorized = processor(
            text=batch_texts,
            images=batch_images,
            padding=True,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs=model.generate(**inputs_dict_tensorized,max_new_tokens=max_new_tokens,temperature=temperature)

        print("🚀 模型生成结束") 

    except Exception as inference_err:
        err_msg=f"[Inference Error]:{inference_err}"
         
        logger.error(err_msg)  
         
        # 实时显示：
        print(err_msg)
         
        return["⚠️ 推理过程中出现异常！"]*len(page_numbers)


    prompt_lengths=[len(processor.tokenizer(txt)["input_ids"])if txt else 0 for txt in batch_texts]

    decoded_results=[]
    
    for idx,out_seq in enumerate(outputs):
        prompt_len=min(prompt_lengths[idx],len(out_seq))
       
        new_token_ids_for_output_slice_based_on_input_length_calc_above = out_seq[prompt_len:]
       
        single_decoded_output_per_page_final = processor.tokenizer.decode(
            new_token_ids_for_output_slice_based_on_input_length_calc_above,
            skip_special_tokens=True).strip()

        if not single_decoded_output_per_page_final.strip():
            single_decoded_output_per_page_final="⚠️ 本页内容解析为空，请检查输入"

        decoded_results.append(single_decoded_output_per_page_final)

    return decoded_results


def process_entire_pdf(pdf_path, batch_size: int = 1): 
    original_file_name = os.path.splitext(os.path.basename(pdf_path))[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    output_dir = "/app/output_results"
    os.makedirs(output_dir, exist_ok=True)  # 自动创建目标目录

    result_txt_path=os.path.join(
        output_dir,
        f"{original_file_name}_{timestamp}.txt"
    )


    pdf_reader_instance_properly_named_concisely = PdfReader(pdf_path)       
    num_pages = len(pdf_reader_instance_properly_named_concisely.pages)

    header = f"📘 PDF文件名称：{original_file_name}.pdf，共 {num_pages} 页\n"

    # 直接打开文件准备实时逐页保存
    with open(result_txt_path, "w", encoding="utf-8") as out_f_handle:
        
        logger.info(header)
        out_f_handle.write(header)
        
        pages_range_all = list(range(1, num_pages + 1))

        for start_idx in range(0, len(pages_range_all), batch_size):
            end_idx = min(start_idx + batch_size, len(pages_range_all))
            current_group_of_pages_processing_in_loop = pages_range_all[start_idx:end_idx]

            logger.info(f"[Processing Batch] Pages:{current_group_of_pages_processing_in_loop}")

            try:
                batched_outputs_gpu_call_efficient_once_only=process_pdf_batch(
                    pdf_path,
                    current_group_of_pages_processing_in_loop
                )

                for relative_index, page_num in enumerate(current_group_of_pages_processing_in_loop):
                    formatted_page_result_str = (
                        f"\n==== 第 {page_num}/{num_pages} 页分析结果 ====\n"
                        f"{batched_outputs_gpu_call_efficient_once_only[relative_index]}\n")

                    # 实时逐页（或逐批）写入到文件而非临时缓存！
                    out_f_handle.write(formatted_page_result_str)

                    # 新增：控制台实时显示每页处理情况
                    print(f"✅ 已完成文件「{original_file_name}.pdf」第 {page_num}/{num_pages} 页的处理")

                # 写完每个batch之后主动清理内存和GPU缓存：
                torch.cuda.empty_cache()
                import gc; gc.collect()

            except Exception as err:
                error_detail=f"[Pages {current_group_of_pages_processing_in_loop}] Error: {str(err)}"
                
                logger.error(error_detail)
                
                for failed_page in current_group_of_pages_processing_in_loop:
                    formatted_error_str=(
                        f"\n==== 第 {failed_page}/{num_pages} 页分析结果 ====\n⚠️ 本页内容解析错误，请手动检查！\n"
                    )
                    
                    out_f_handle.write(formatted_error_str)

    completion_message=f"\n✅ Done! 分析完毕已保存为: 「{result_txt_path}」\n"

    print(completion_message)
    logger.info(completion_message)            


# 新增函数：自动遍历指定目录及其下所有子目录，并返回全部pdf文件路径列表。
def find_all_pdfs(root_dir='./pdf'):
    pdf_files_list = []

    for root, directories, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(".pdf"):
                fullpath = os.path.join(root, filename)
                pdf_files_list.append(fullpath)

    return sorted(pdf_files_list)



if __name__ == "__main__":
    try:
        all_pdf_paths = find_all_pdfs('./pdf')

        print(f"一共找到 {len(all_pdf_paths)} 个 PDF 待处理，开始批量执行...")

        for pdf_path in all_pdf_paths:
            process_entire_pdf(pdf_path, batch_size=1)  # 可调整batch大小

    except Exception as main_err:
        error_main_run = f"[Main Error] 程序运行时发生错误，请检查路径或环境配置是否正确! 错误详情: {main_err}"
        
        print(error_main_run)


