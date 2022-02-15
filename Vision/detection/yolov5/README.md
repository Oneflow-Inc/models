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