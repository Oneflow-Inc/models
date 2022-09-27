from DenseNet import DenseNet161_pre
import flowvision.transforms as ft
import oneflow as of
import cv2
import os
import json

device = 'cuda'
file = open('submit_example.json')
infos = json.load(file)
annotations = infos['annotations']

size = 224
transforms = ft.Compose([
            ft.ToTensor(),
            ft.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
# I use four models for voting, line 21-line 40 is replaced by your stored weight path
model1 = DenseNet161_pre(num_classes=10, pretrained=False)
model2 = DenseNet161_pre(num_classes=10, pretrained=False)
model3 = DenseNet161_pre(num_classes=10, pretrained=False)
model4 = DenseNet161_pre(num_classes=10, pretrained=False)

model1.load_state_dict(of.load(''))
model1.to(device)
model1.eval()

model2.load_state_dict(of.load(''))
model2.to(device)
model2.eval()

model3.load_state_dict(of.load(''))
model3.to(device)
model3.eval()

model4.load_state_dict(of.load(''))
model4.to(device)
model4.eval()

result = {}
ann = []
ii = 0
nn1 = 0
# working with dict
def f(label1):
    temp = -1
    re = []
    for k in label1.keys():
        if label1[k] > temp:
            re = []
            temp = label1[k]
            re.append(k)
        elif label1[k] == temp:
            re.append(k)
        else:
            pass
    return re


for temp in annotations:
    print(ii)
    ii += 1
    label1 = {}
    label2 = []
    imgfile = temp['filename']
    img = cv2.imread(imgfile)
    img = cv2.blur(img, ksize=(9, 9))
    # Five pictures, I only got five pictures, you can continue to add
    img1 = cv2.rotate(img, 2)
    img1 = transforms(img1)
    img1 = of.reshape(img1, (1, 3, size, size))
    img1 = img1.to('cuda')
    #
    img2 = cv2.rotate(img, 0)
    img2 = transforms(img2)
    img2 = of.reshape(img2, (1, 3, size, size))
    img2 = img2.to('cuda')
    #
    img3 = cv2.rotate(img, 1)
    img3 = transforms(img3)
    img3 = of.reshape(img3, (1, 3, size, size))
    img3 = img3.to('cuda')
    #
    img4 = transforms(img)
    img4 = of.reshape(img4, (1, 3, size, size))
    img4 = img4.to('cuda')
    #
    img5 = cv2.flip(img, 0)
    img5 = transforms(img5)
    img5 = of.reshape(img5, (1, 3, size, size))
    img5 = img5.to('cuda')
    input1 = of.cat([img1, img2, img3, img4, img5], dim=0)
    # Generate twenty results
    with of.no_grad():
        #
        pred1 = model1(input1)
        _, indices1 = of.topk(pred1, k=1, dim=1)  # 5 * 1
        for index in indices1:
            label2.append(index.item())
        #
        pred2 = model2(input1)
        _, indices2 = of.topk(pred2, k=1, dim=1)  # 5 * 1
        for index in indices2:
            label2.append(index.item())
        #
        pred3 = model3(input1)
        _, indices3 = of.topk(pred3, k=1, dim=1)  # 5 * 1
        for index in indices3:
            label2.append(index.item())
        #
        pred4 = model4(input1)
        _, indices4 = of.topk(pred4, k=1, dim=1)  # 5 * 1
        for index in indices4:
            label2.append(index.item())
    #
    for re in label2:
        if re in label1.keys():
            label1[re] += 1
        else:
            label1[re] = 1

    re1 = f(label1)

    if len(re1) == 1:
        ann.append({'filename': imgfile, 'label': int(re1[0])})
    else:
        print('model4')
        nn1 += 1
        re1 = indices4[3, 0]
        ann.append({'filename': imgfile, 'label': int(re1)})


result['annotations'] = ann
# warnning
if os.path.exists('bestmodel.json'):
    os.remove('bestmodel.json')
with open("bestmodel.json", "w", encoding='utf-8') as f:  # 设置'utf-8'编码
    f.write(json.dumps(result, ensure_ascii=False, indent=4))
