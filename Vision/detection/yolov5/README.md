### 准备模型权重
models/Vision/detection/yolov5是oneflow版本yolov5的工作目录，models/Vision/detection/yolov5/yolov5_torch是torch版本yolov5的工作目录
##### oneflow
从路径oss://oneflow-test/yolov5-nsys下载oneflow权重yolov5_ckpt（文件较多，所以没有获取地址），保存路径如下
```
models/
    Vision/
        detection/
            yolov5/
                yolov5_ckpt/
```
##### torch
权重下载链接https://oneflow-test.oss-cn-beijing.aliyuncs.com/yolov5-nsys/yolov5s.pt，保存路径如下
```
models/
    Vision/
        detection/
            yolov5/
               yolov5_torch/
                   yolov5s.pt 
```
### 运行
```
cd models/Vision/detection/yolov5
pip install -r requirements.txt
```
##### 生成oneflow的profile
```
/usr/local/cuda-11.2/nsight-systems-2020.4.3/target-linux-x64/nsys profile --stats=true \
--output "yolov5-oneflow-infer" \
python simple_infer.py
```
##### 生成torch的profile
```
cd ./yolov5_torch
```
```
/usr/local/cuda-11.2/nsight-systems-2020.4.3/target-linux-x64/nsys profile --stats=true \
--output "yolov5-torch-infer" \
python simple_infer.py
```