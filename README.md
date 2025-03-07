## Docker One-Click Setup for olmocr

Based on the project [`https://github.com/allenai/olmocr`](https://github.com/allenai/olmocr).

After trying several projects' knowledge base functionalities, I found that directly processing PDFs with OpenAI's vector embedding models gives average or poor results. Some content such as mathematical formulas and charts could even fail to be processed entirely. However, converting these contents first into text before feeding them into embedding models should significantly improve accuracy.

## Features

Processes all PDF files within a specified local directory and saves results locally; supports concurrent processing of multi-page PDFs; continues processing subsequent pages even if some pages fail.

1. 【Recommended】`process_pdfs.py` is suitable for handling large numbers of files simultaneously. It will process all PDF files in the current folder (including subfolders), saving results as `{{original_filename}}.txt` under the folder `./output_results`. This document demonstrates using this method.

2. 【Not recommended】Running `appE.py` provides a web frontend page—it's advised to access it via IP + port directly. Supports uploading multiple files at once; upon completing processing, the frontend will display a completion message, and results are saved in the same directory as appE.py. Multiple uploaded files are processed sequentially by each page in order.

## Example Results:

### Processing screenshots:

![Processing example 1](https://cdn.jsdelivr.net/gh/klearcc/pic/img202503071720515.png)

![Processing example 2](https://cdn.jsdelivr.net/gh/klearcc/pic/img202503071742796.png)

### Output examples:

Math formula recognition is excellent, but images aren't included yet.

![Result example 1](https://cdn.jsdelivr.net/gh/klearcc/pic/img202503071721272.png)

![Result example 2](https://cdn.jsdelivr.net/gh/klearcc/pic/img202503071721448.png)



## Environment Requirements

Ubuntu OS with GPU support: GPU memory >20GB, disk space around hundreds GB.

---

## Usage Instructions

1\. Pull image (~20GB):

```bash
docker pull caldedaniele/olmocr-app
```

2\. Create container instance:

```bash
docker run -d \
  --gpus all \
  --name olmocr-app-container \
  -p 80:7860 \
  caldedaniele/olmocr-app sleep infinity
```

3\. Enter container and install dependencies: 

```bash
docker exec -it olmocr-app-container bash

# Install necessary dependencies
apt update
apt install -y gcc build-essential python3-dev libffi-dev libssl-dev libcurl4-openssl-dev

# Solve issues related to displaying Chinese characters on server:
sudo apt update
sudo apt install -y locales
sudo locale-gen en_US.UTF-8
sudo dpkg-reconfigure locales
echo "export LANG=en_US.UTF-8" >> ~/.bashrc
echo "export LC_ALL=en_US.UTF-8" >> ~/.bashrc
source ~/.bashrc 
```

4\. Execute scripts inside container:

```bash 
# Create folder for PDFs awaiting processing:
mkdir pdf  

# Download script designed for bulk-processing numerous files at once.
wget https://github.com/Klearcc/olmocr-oneclick/blob/main/process_pdfs.py  

# First execution downloads approximately ~20GB worth additional data/files.
python process_pdfs.py  
```

Upon success you'll see similar output information like below — after seeing highlighted arrow outputs you must wait patiently until model loading into GPU finishes (this may take significant time):

![Successful setup output sample](https://cdn.jsdelivr.net/gh/klearcc/pic/img202503071748661.png)

Useful commands assistance: 

```bash 
# Background execution command-line prompt:
nohup python process_pdfs.py > output.log 2>&1 &   

# Real-time log monitoring:
tail -f output.log  

# Continuously monitor GPU status ensuring GPUs properly working without full-load/OOM situations occurring:
watch -n0.5 nvidia-smi  
```

---

## Customization guide 

The parameter `batch_size` indicates how many PDF pages get concurrently processed—the default setting currently equals one (`batch_size=1`). You can increase it depending upon your system resource capability—pay close attention regarding available memory size limitations! 

You also can customize location settings defining target folders containing documents needing conversion by modifying source-code line specifying path settings explicitly defined here within script code snippet below (`all_pdf_paths = find_all_pdfs('./pdf')`).

---

## References & Further Reading Sources 📖 :

[`Original olmocr GitHub Repository`](https://github.com/allenai/olmocr)   
[`Issue Discussion Link #75`](https://github.com/allenai/olmocr/issues/75#issue-2889882053)
