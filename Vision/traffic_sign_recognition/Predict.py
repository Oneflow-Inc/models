from DenseNet import DenseNet161_pre
import flowvision.transforms as ft
import cv2
import json
import oneflow as of

pth_path = ''
model = DenseNet161_pre(num_classes=10, pretrained=False)
model.load_state_dict(of.load(pth_path))
model.to('cuda')
model.eval()

file = open('submit_example.json')
infos = json.load(file)
annotations = infos['annotations']

size = 224
transforms = ft.Compose([
            ft.ToTensor(),
            ft.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )
result = {}
ann = []
len1 = len(annotations)
print('The total number:', len1)
for temp in annotations:
    imgfile = temp['filename']
    img = cv2.imread(imgfile)
    img = cv2.blur(img, ksize=(9, 9))
    img = transforms(img)
    img = of.reshape(img, (1, 3, size, size))
    img = img.to('cuda')
    pred = model(img)
    _, indices = of.topk(pred, k=1, dim=1)
    re = indices.item()
    ann.append({'filename': imgfile, 'label': int(re)})

result['annotations'] = ann
with open("{}.json".format(pth_path), "w", encoding='utf-8') as f:  # 设置'utf-8'编码
    f.write(json.dumps(result, ensure_ascii=False, indent=4))
