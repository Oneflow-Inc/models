### 准备模型权重
```
从路径oss://oneflow-test/yolov5-nsys下载oneflow权重yolov5_ckpt（文件较多，所以没有获取地址），保存路径如下
models/
    Vision/
        detection/
            yolov5/
               yolov5_ckpt/ 
```
### 运行
```
pip install -r requirements.txt
```
```
/usr/local/cuda-11.2/nsight-systems-2020.4.3/target-linux-x64/nsys profile --stats=true \
--output "yolov5" \
python simple_infer.py
```