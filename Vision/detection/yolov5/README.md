### 运行
```
pip install -r requirements.txt
```
```
/usr/local/cuda-11.2/nsight-systems-2020.4.3/target-linux-x64/nsys profile --stats=true \
--output "yolov5" \
python simple_infer.py
```