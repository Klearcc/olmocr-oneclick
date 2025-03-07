## docker一键搭建olmocr

基于https://github.com/allenai/olmocr项目。

## 功能



## 环境

ubuntu带GPU。显存>20G，硬盘百G

## 使用

1. image20多个G

    ```
    docker pull  caldedaniele/olmocr-app
    ```

2. 创建容器

3. ```
    docker run -d \
      --gpus all \
      --name olmocr-app-container \
      -p 80:7860 \
      caldedaniele/olmocr-app sleep infinity
    ```

3. 进容器添加依赖 + 修改文件 + 启动

4. ```
    docker exec -it olmocr-app-container bash
    
    apt update
    apt install -y gcc build-essential python3-dev libffi-dev libssl-dev libcurl4-openssl-dev
    
    
    ```

5. 







## 参考

https://github.com/allenai/olmocr

https://github.com/allenai/olmocr/issues/75#issue-2889882053

