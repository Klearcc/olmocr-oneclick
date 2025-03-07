## docker一键搭建olmocr

基于`https://github.com/allenai/olmocr`项目。

尝试了多个项目的知识库功能，发现openai的几个向量模型对pdf直接处理后的效果一般，甚至有些无法向量化。比如pdf内的数学公式、图表等内容。如果对这些内容先转成文本再用向量模型去处理，准确性应该会大大提高。

## 功能

处理本地特定文件夹内的所有pdf文件并将处理结果保存至本地；可同时对多页pdf处理；某一页处理失败后继续处理下一页。

1、【推荐】`process_pdfs.py`适合处理大量文件时的情况。其会处理当前pdf文件夹（包括子文件夹）内所有pdf文件，结果以``{{原文件名}}.txt`保存在`./output_results`文件夹内。本文均用此方式进行演示。

2、【不推荐】`appE.py`运行后会有一个前端页面，建议直接用ip+端口访问。支持多文件上传，处理结束后前端会提示结束信息，结果会保存在appE文件的同级目录下。上传多个文件时后台会按照顺序对每个pdf的每一页进行处理并保存至文件。



## 效果如下

处理过程：

![image-20250307172019600](https://cdn.jsdelivr.net/gh/klearcc/pic/img202503071720515.png)

![image-20250307174217746](https://cdn.jsdelivr.net/gh/klearcc/pic/img202503071742796.png)

处理结果：

对数学公式的处理很不错，但还没贴图。

![image-20250307172118225](https://cdn.jsdelivr.net/gh/klearcc/pic/img202503071721272.png)

![image-20250307172140399](https://cdn.jsdelivr.net/gh/klearcc/pic/img202503071721448.png)



## 环境

ubuntu带GPU。显存>20G，硬盘百G

## 使用

1、image20多个G

```
docker pull  caldedaniele/olmocr-app
```

2、创建容器

```
docker run -d \
  --gpus all \
  --name olmocr-app-container \
  -p 80:7860 \
  caldedaniele/olmocr-app sleep infinity
```

3、 进容器添加依赖

```
docker exec -it olmocr-app-container bash

## 安装所需依赖
apt update
apt install -y gcc build-essential python3-dev libffi-dev libssl-dev libcurl4-openssl-dev

## 解决服务器内无法显示中文的问题
sudo apt update
sudo apt install -y locales
sudo locale-gen en_US.UTF-8
sudo dpkg-reconfigure locales
echo "export LANG=en_US.UTF-8" >> ~/.bashrc
echo "export LC_ALL=en_US.UTF-8" >> ~/.bashrc
source ~/.bashrc
```

4、 运行脚本

```
## 创建存放待处理的pdf文件和文件夹
mkdir pdf

## 下载脚本，此脚本适合一次性处理大量文件。
wget https://github.com/Klearcc/olmocr-oneclick/blob/main/process_pdfs.py

## 第一次执行会下载20G左右的文件。
python process_pdfs.py

```

成功后会看到类似下面的提示。出现箭头处的输出后需要等待模型加载至GPU，可能需要较长时间。

![image-20250307174851608](https://cdn.jsdelivr.net/gh/klearcc/pic/img202503071748661.png)

命令辅助

```
## 后台挂起
nohup python process_pdfs.py > output.log 2>&1 &
## 实时查看日志
tail -f output.log

## 查看实时gpu状态确保gpu确实工作中且未满载OOM
watch -n0.5 nvidia-smi 


```

## 自定义

batch_size代表同时处理pdf的页数，当前默认为1，如果机器性能好的话可以调大。注意显存的大小。

文件内定义待处理的文件夹的位置的代码为`all_pdf_paths = find_all_pdfs('./pdf')`，可以自行调整。





## 参考

https://github.com/allenai/olmocr

https://github.com/allenai/olmocr/issues/75#issue-2889882053

